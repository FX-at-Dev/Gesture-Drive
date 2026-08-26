import cv2
import mediapipe as mp
import math
import time
import platform
from pynput.keyboard import Controller, Listener

from config import load_config, resolve_key
from tray_icon import TrayIcon

cfg = load_config()

KB = cfg["keybindings"]
CONTROL_ACTIONS = ["steer_left", "steer_right", "accel", "brake", "reverse", "handbrake", "gear_up", "gear_down", "game_pause", "confirm"]
KEYMAP = {action: resolve_key(KB[action]) for action in CONTROL_ACTIONS}
PAUSE_KEY = resolve_key(KB["pause"])

FLIP_CAMERA = cfg["flip_camera"]

PALETTES = {
    "dark": {
        "WHEEL": (80, 200, 255), "LEFT": (60, 120, 255), "RIGHT": (50, 220, 140),
        "NEUTRAL": (200, 200, 200), "TEXT": (255, 255, 255), "ACCENT": (0, 180, 255),
        "HAND_L": (255, 130, 60), "HAND_R": (60, 230, 130), "ACCEL": (50, 220, 100),
        "BRAKE": (0, 60, 255), "REVERSE": (220, 80, 220), "HANDBRAKE": (0, 210, 255), "BG": (10, 10, 20),
        "LANDMARK": (200, 200, 255), "CONNECTION": (80, 80, 100),
    },
    "light": {
        "WHEEL": (0, 120, 200), "LEFT": (30, 80, 200), "RIGHT": (20, 140, 90),
        "NEUTRAL": (120, 120, 120), "TEXT": (20, 20, 20), "ACCENT": (0, 110, 190),
        "HAND_L": (200, 90, 20), "HAND_R": (20, 140, 80), "ACCEL": (20, 140, 60),
        "BRAKE": (190, 20, 20), "REVERSE": (170, 40, 170), "HANDBRAKE": (0, 130, 190), "BG": (225, 225, 232),
        "LANDMARK": (60, 60, 90), "CONNECTION": (150, 150, 170),
    },
}

keyboard   = Controller()
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def is_open_hand(hand_landmarks):
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    extended = sum(
        1 for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    )
    return extended >= cfg["detection"]["open_finger_thresh"]


THUMB_EXTEND_DIST = 0.09
THUMB_ANGLE_MARGIN = 0.05


def _thumb_extended_away(lm):
    tip, index_mcp = lm.landmark[4], lm.landmark[5]
    return math.hypot(tip.x - index_mcp.x, tip.y - index_mcp.y) > THUMB_EXTEND_DIST


def is_thumbs_up(hand_landmarks):
    lm = hand_landmarks.landmark
    tip, ip, mcp = lm[4], lm[3], lm[2]
    pointing_up = (mcp.y - tip.y) > THUMB_ANGLE_MARGIN and (ip.y - tip.y) > 0.02
    return (
        pointing_up
        and _thumb_extended_away(hand_landmarks)
        and not is_open_hand(hand_landmarks)
    )


def both_thumbs_up(left_lm, right_lm):
    return left_lm is not None and right_lm is not None and is_thumbs_up(left_lm) and is_thumbs_up(right_lm)


class SteeringController:
    def __init__(self, cfg, keymap):
        self.keymap = keymap
        self.keys_held = {a: False for a in ["steer_left", "steer_right", "accel", "brake", "reverse", "handbrake"]}

        self.alpha = cfg["steering"]["smoothing_alpha"]
        self.dead_zone = cfg["steering"]["dead_zone_deg"]
        self.release_zone = cfg["steering"]["release_zone_deg"]
        self.soft_zone = cfg["steering"]["soft_zone_deg"]

        self.pwm_period = max(cfg["throttle"]["pwm_period_ms"], 1) / 1000.0
        self.min_intensity_y = cfg["throttle"]["min_intensity_y"]
        self.max_intensity_y = cfg["throttle"]["max_intensity_y"]

        self.handbrake_enabled = cfg["handbrake"]["enabled"]
        self.handbrake_dist = cfg["handbrake"]["distance_thresh"]

        self.game_pause_enabled = cfg["game_pause"]["enabled"]
        self.game_pause_dist = cfg["game_pause"]["distance_thresh"]
        self.game_pause_debounce = cfg["game_pause"]["debounce_ms"] / 1000.0
        self.last_game_pause_time = 0.0

        self.confirm_enabled = cfg["confirm"]["enabled"]
        self.confirm_debounce = cfg["confirm"]["debounce_ms"] / 1000.0
        self.last_confirm_time = 0.0

        self.gear_enabled = cfg["gear_shift"]["enabled"]
        self.gear_debounce = cfg["gear_shift"]["debounce_ms"] / 1000.0
        self.min_gear = cfg["gear_shift"]["min_gear"]  # 0 = Reverse
        self.max_gear = cfg["gear_shift"]["max_gear"]
        self.gear_mode = cfg["gear_shift"]["start_mode"]  # "manual" or "automatic"
        self.gear = 1
        self.last_gear_shift_time = 0.0

        self.smoothed_angle = 0.0
        self._angle_init = False
        self.start_time = time.time()

    def _press(self, action):
        if not self.keys_held[action]:
            keyboard.press(self.keymap[action])
            self.keys_held[action] = True

    def _release(self, action):
        if self.keys_held[action]:
            keyboard.release(self.keymap[action])
            self.keys_held[action] = False

    def _tap(self, action):
        keyboard.press(self.keymap[action])
        keyboard.release(self.keymap[action])

    def release_all(self):
        for action in list(self.keys_held.keys()):
            try:
                keyboard.release(self.keymap[action])
            except Exception:
                pass
            self.keys_held[action] = False
        self._angle_init = False
        self.smoothed_angle = 0.0

    def smooth_angle(self, raw_angle):
        if self._angle_init:
            self.smoothed_angle = self.alpha * raw_angle + (1 - self.alpha) * self.smoothed_angle
        else:
            self.smoothed_angle = raw_angle
            self._angle_init = True
        return self.smoothed_angle

    def update_steer(self, left_wrist, right_wrist):
        dx = right_wrist[0] - left_wrist[0]
        dy = right_wrist[1] - left_wrist[1]

        raw_angle_deg = math.degrees(math.atan2(dy, dx))
        angle = self.smooth_angle(raw_angle_deg)

        direction = "STRAIGHT"
        if angle < -self.dead_zone:
            direction = "LEFT"
        elif angle > self.dead_zone:
            direction = "RIGHT"
        elif self.keys_held["steer_left"] and angle > -self.release_zone:
            direction = "STRAIGHT"
        elif self.keys_held["steer_right"] and angle < self.release_zone:
            direction = "STRAIGHT"

        strength = 0.0
        if direction == "LEFT":
            strength = min(1.0, (abs(angle) - self.dead_zone) / (self.soft_zone - self.dead_zone))
            self._press("steer_left")
            self._release("steer_right")
        elif direction == "RIGHT":
            strength = min(1.0, (abs(angle) - self.dead_zone) / (self.soft_zone - self.dead_zone))
            self._press("steer_right")
            self._release("steer_left")
        else:
            self._release("steer_left")
            self._release("steer_right")

        return angle, direction, strength

    def _height_intensity(self, avg_y):
        lo, hi = self.min_intensity_y, self.max_intensity_y
        if hi <= lo:
            return 1.0
        return max(0.0, min(1.0, (hi - avg_y) / (hi - lo)))

    def _set_pwm(self, action, intensity, now):
        if intensity >= 0.97:
            self._press(action)
        elif intensity <= 0.03:
            self._release(action)
        else:
            phase = (now - self.start_time) % self.pwm_period
            if phase < self.pwm_period * intensity:
                self._press(action)
            else:
                self._release(action)

    def update_throttle(self, left_open, right_open, avg_wrist_y, hand_dist, now):
        both_open = left_open and right_open
        both_fist = (not left_open) and (not right_open)

        if self.handbrake_enabled and both_fist and hand_dist < self.handbrake_dist:
            self._release("accel")
            self._release("brake")
            self._release("reverse")
            self._press("handbrake")
            return "HANDBRAKE", 1.0

        if self.game_pause_enabled and both_open and hand_dist < self.game_pause_dist:
            self._release("accel")
            self._release("brake")
            self._release("reverse")
            self._release("handbrake")
            if (now - self.last_game_pause_time) >= self.game_pause_debounce:
                self._tap("game_pause")
                self.last_game_pause_time = now
            return "GAME_PAUSE", 1.0

        self._release("handbrake")
        intensity = self._height_intensity(avg_wrist_y)
        in_reverse = self.gear == 0

        if both_fist:
            if in_reverse:
                self._set_pwm("reverse", intensity, now)
                self._release("accel")
            else:
                self._set_pwm("accel", intensity, now)
                self._release("reverse")
            self._release("brake")
            return ("REVERSE" if in_reverse else "ACCEL"), intensity
        elif both_open:
            self._set_pwm("brake", intensity, now)
            self._release("accel")
            self._release("reverse")
            return "BRAKE", intensity
        else:
            self._release("accel")
            self._release("brake")
            self._release("reverse")
            return "NEUTRAL", 0.0

    def toggle_gear_mode(self):
        self.gear_mode = "automatic" if self.gear_mode == "manual" else "manual"
        # Automatic only ever shows Reverse or Drive - collapse any numbered gear to "D".
        if self.gear_mode == "automatic" and self.gear != 0:
            self.gear = 1

    def update_confirm(self, now):
        if not self.confirm_enabled or (now - self.last_confirm_time) < self.confirm_debounce:
            return False
        self._tap("confirm")
        self.last_confirm_time = now
        return True

    def update_gear_shift(self, left_lm, right_lm, now):
        """Right-hand thumbs-up shifts up; left-hand thumbs-up shifts down.
        Which hand does the thumbs-up decides the direction, so both shifts use
        the same natural gesture instead of requiring an awkward thumbs-down.
        """
        if not self.gear_enabled or (now - self.last_gear_shift_time) < self.gear_debounce:
            return None

        if right_lm is not None and is_thumbs_up(right_lm):
            self._tap("gear_up")
            self.gear = 1 if self.gear_mode == "automatic" else min(self.max_gear, self.gear + 1)
            self.last_gear_shift_time = now
            return "UP"

        if left_lm is not None and is_thumbs_up(left_lm):
            self._tap("gear_down")
            self.gear = 0 if self.gear_mode == "automatic" else max(self.min_gear, self.gear - 1)
            self.last_gear_shift_time = now
            return "DOWN"

        return None


def draw_steering_wheel(frame, pal, center, angle_deg, direction, strength):
    h, w = frame.shape[:2]
    radius = int(min(w, h) * 0.10)
    cx, cy = center

    color = pal["NEUTRAL"]
    if direction == "LEFT":
        color = pal["LEFT"]
    elif direction == "RIGHT":
        color = pal["RIGHT"]

    cv2.circle(frame, (cx + 3, cy + 3), radius, (0, 0, 0), 4)
    cv2.circle(frame, (cx, cy), radius, color, 3)

    for sa in [0, 120, 240]:
        rad = math.radians(sa - angle_deg)
        x1 = int(cx + radius * 0.4 * math.cos(rad))
        y1 = int(cy - radius * 0.4 * math.sin(rad))
        x2 = int(cx + radius * 0.95 * math.cos(rad))
        y2 = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.circle(frame, (cx, cy), 6, color, -1)

    if direction != "STRAIGHT":
        start_a = -30 if direction == "RIGHT" else 150
        end_a   =  30 if direction == "RIGHT" else 210
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_a, end_a, color, 5)


def draw_hud(frame, pal, angle, direction, strength, throttle_mode, throttle_strength,
             gear, gear_mode, both_hands_visible, left_open, right_open, fps, show_angle):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 160), (w, h), pal["BG"], -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bar_w = int(w * 0.5)
    bar_h = 14
    bar_x = (w - bar_w) // 2
    bar_y = h - 110
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

    mid = bar_x + bar_w // 2
    cv2.rectangle(frame, (mid - 2, bar_y - 4), (mid + 2, bar_y + bar_h + 4), (180, 180, 180), -1)

    fill_len = int((bar_w // 2) * strength)
    if direction == "LEFT" and fill_len > 0:
        cv2.rectangle(frame, (mid - fill_len, bar_y), (mid, bar_y + bar_h), pal["LEFT"], -1)
    elif direction == "RIGHT" and fill_len > 0:
        cv2.rectangle(frame, (mid, bar_y), (mid + fill_len, bar_y + bar_h), pal["RIGHT"], -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    dir_color = pal["LEFT"] if direction == "LEFT" else (pal["RIGHT"] if direction == "RIGHT" else pal["NEUTRAL"])
    cv2.putText(frame, "<- LEFT",  (bar_x, bar_y - 10),               font, 0.45, pal["LEFT"],  1)
    cv2.putText(frame, "RIGHT ->", (bar_x + bar_w - 80, bar_y - 10),  font, 0.45, pal["RIGHT"], 1)
    cv2.putText(frame, direction,  (mid - 30, bar_y + bar_h + 28),    font, 0.8,  dir_color, 2)

    if show_angle:
        cv2.putText(frame, f"{angle:+.1f} deg", (bar_x, h - 80), font, 0.55, pal["TEXT"], 1)

    throttle_colors = {
        "ACCEL": pal["ACCEL"], "BRAKE": pal["BRAKE"], "REVERSE": pal["REVERSE"],
        "HANDBRAKE": pal["HANDBRAKE"], "GAME_PAUSE": pal["ACCENT"], "NEUTRAL": pal["NEUTRAL"],
    }
    throttle_labels = {
        "ACCEL":      f"ACCEL [UP] {int(throttle_strength * 100)}%",
        "BRAKE":      f"BRAKE [DOWN] {int(throttle_strength * 100)}%",
        "REVERSE":    f"REVERSE [UP] {int(throttle_strength * 100)}%",
        "HANDBRAKE":  "HANDBRAKE [SPACE]",
        "GAME_PAUSE": "GAME PAUSE [ESC]",
        "NEUTRAL":    "NEUTRAL",
    }
    throttle_color = throttle_colors[throttle_mode]
    throttle_label = throttle_labels[throttle_mode]

    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), (30, 30, 40), -1)
    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), throttle_color, 2)
    cv2.putText(frame, throttle_label, (bar_x + 10, h - 48), font, 0.6, throttle_color, 2)

    # Gear/mode badge gets its own box in the panel's left margin (otherwise empty)
    # so it never overlaps the throttle label, which can run long (e.g. "REVERSE [UP] 100%").
    gear_text  = "R" if gear == 0 else ("D" if gear_mode == "automatic" else str(gear))
    mode_tag   = "AUTO" if gear_mode == "automatic" else "MANUAL"
    gear_box_x = max(10, bar_x - 150)
    cv2.rectangle(frame, (gear_box_x, h - 65), (bar_x - 10, h - 42), (30, 30, 40), -1)
    cv2.rectangle(frame, (gear_box_x, h - 65), (bar_x - 10, h - 42), pal["ACCENT"], 2)
    cv2.putText(frame, f"GEAR {gear_text}", (gear_box_x + 10, h - 48), font, 0.6, pal["ACCENT"], 2)
    cv2.putText(frame, mode_tag, (gear_box_x + 10, h - 72), font, 0.42, pal["TEXT"], 1)

    l_label = "OPEN" if left_open else "FIST"
    r_label = "OPEN" if right_open else "FIST"
    l_color = pal["BRAKE"] if left_open else pal["ACCEL"]
    r_color = pal["BRAKE"] if right_open else pal["ACCEL"]
    cv2.putText(frame, f"L:{l_label}", (bar_x + bar_w + 10, h - 100), font, 0.5, l_color, 1)
    cv2.putText(frame, f"R:{r_label}", (bar_x + bar_w + 10, h - 80),  font, 0.5, r_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 30), font, 0.55, pal["ACCENT"], 1)

    status       = "BOTH HANDS DETECTED" if both_hands_visible else "SHOW BOTH HANDS"
    status_color = (60, 220, 60) if both_hands_visible else (0, 80, 255)
    cv2.putText(frame, status, (10, 30), font, 0.55, status_color, 1)

    draw_steering_wheel(frame, pal, (w - 80, h - 80), angle, direction, strength)


def draw_hand_connection(frame, pal, lw, rw):
    lx, ly = lw
    rx, ry = rw
    cv2.line(frame, (lx, ly), (rx, ry), (30, 100, 200), 8)
    cv2.line(frame, (lx, ly), (rx, ry), pal["ACCENT"], 2)
    cv2.circle(frame, (lx, ly), 10, pal["HAND_L"], -1)
    cv2.circle(frame, (rx, ry), 10, pal["HAND_R"], -1)
    cv2.circle(frame, (lx, ly), 13, pal["HAND_L"], 2)
    cv2.circle(frame, (rx, ry), 13, pal["HAND_R"], 2)
    mx = (lx + rx) // 2
    my = (ly + ry) // 2
    cv2.circle(frame, (mx, my), 7, pal["WHEEL"], -1)


def draw_legend(frame, pal):
    lines = [
        ("GESTURES", True),
        ("Fist (both hands) = Accelerate  (or Reverse, in gear R)", False),
        ("Open palm (both hands) = Brake", False),
        ("Fists brought together = Handbrake", False),
        ("Open palms brought together = In-game Pause (Esc)", False),
        ("Both hands thumbs up = Confirm (Enter)", False),
        ("RIGHT hand thumbs up = Gear up / Drive", False),
        ("LEFT hand thumbs up = Gear down / Reverse", False),
        ("Tilt hands left/right = Steer", False),
        ("", False),
        ("KEYS", True),
        (f"[{KB['pause'].upper()}] Pause / resume (works unfocused)", False),
        (f"[{KB['gear_mode_toggle'].upper()}] Toggle manual / automatic gears", False),
        (f"[{KB['theme_toggle'].upper()}] Toggle light / dark theme", False),
        (f"[{KB['legend'].upper()}] Toggle this help", False),
        (f"[{KB['quit'].upper()}] Quit", False),
    ]
    x0, y0 = 10, 55
    box_h = 22 * len(lines) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 10, y0 - 20), (x0 + 380, y0 - 20 + box_h), pal["BG"], -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    for i, (line, heading) in enumerate(lines):
        color = pal["ACCENT"] if heading else pal["TEXT"]
        weight = 2 if heading else 1
        cv2.putText(frame, line, (x0, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)


def show_setup_screen(cap, seconds=3.0):
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        remaining = max(0.0, seconds - (time.time() - start))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "GESTURE DRIVE", (w // 2 - 170, h // 2 - 90), font, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, "Fist = Accelerate    Open palm = Brake", (w // 2 - 260, h // 2 - 40), font, 0.6, (210, 210, 210), 1)
        cv2.putText(frame, "Tilt both hands left/right to steer", (w // 2 - 220, h // 2 - 10), font, 0.6, (210, 210, 210), 1)
        cv2.putText(frame, "Fists together = Handbrake   R-hand/L-hand thumbs up = Shift up/down", (w // 2 - 320, h // 2 + 20), font, 0.55, (210, 210, 210), 1)
        cv2.putText(frame, "Palms together = Pause (Esc)   Both thumbs up = Confirm (Enter)", (w // 2 - 300, h // 2 + 48), font, 0.55, (210, 210, 210), 1)
        cv2.putText(frame, f"Starting in {remaining:.1f}s  (press any key to skip)", (w // 2 - 220, h // 2 + 90), font, 0.6, (0, 180, 255), 1)

        cv2.imshow("Virtual Steering Wheel", frame)
        key = cv2.waitKey(1) & 0xFF
        if remaining <= 0 or key != 255:
            break


def get_camera_backend():
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    elif system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_camera(index):
    cap = cv2.VideoCapture(index, get_camera_backend())
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def _key_matches(pynput_key, target):
    if isinstance(target, str):
        return getattr(pynput_key, "char", None) == target
    return pynput_key == target


CONTROLS_WINDOW = "Controls"


def setup_trackbars(controller):
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 360, 220)
    nop = lambda x: None
    cv2.createTrackbar("Dead Zone",       CONTROLS_WINDOW, int(controller.dead_zone), 45, nop)
    cv2.createTrackbar("Release Zone",    CONTROLS_WINDOW, int(controller.release_zone), 30, nop)
    cv2.createTrackbar("Soft Zone",       CONTROLS_WINDOW, int(controller.soft_zone), 60, nop)
    cv2.createTrackbar("Smoothing %",     CONTROLS_WINDOW, int(controller.alpha * 100), 100, nop)
    cv2.createTrackbar("Handbrake Dist %", CONTROLS_WINDOW, int(controller.handbrake_dist * 100), 50, nop)


def apply_trackbars(controller):
    controller.dead_zone = cv2.getTrackbarPos("Dead Zone", CONTROLS_WINDOW)
    controller.release_zone = cv2.getTrackbarPos("Release Zone", CONTROLS_WINDOW)
    controller.soft_zone = max(controller.dead_zone + 1, cv2.getTrackbarPos("Soft Zone", CONTROLS_WINDOW))
    controller.alpha = max(0.01, cv2.getTrackbarPos("Smoothing %", CONTROLS_WINDOW) / 100.0)
    controller.handbrake_dist = cv2.getTrackbarPos("Handbrake Dist %", CONTROLS_WINDOW) / 100.0


def main():
    cap = open_camera(cfg["camera_index"])
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("  -> macOS: System Settings > Privacy & Security > Camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    controller = SteeringController(cfg, KEYMAP)
    setup_trackbars(controller)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=cfg["detection"]["min_detection_confidence"],
        min_tracking_confidence=cfg["detection"]["min_tracking_confidence"],
    )

    runtime_state = {"paused": False, "quit": False, "show_legend": False, "theme": cfg["theme"]}

    def on_press(key):
        if _key_matches(key, PAUSE_KEY):
            runtime_state["paused"] = not runtime_state["paused"]

    hotkey_listener = Listener(on_press=on_press)
    hotkey_listener.daemon = True
    hotkey_listener.start()

    tray = TrayIcon(
        on_toggle_pause=lambda: runtime_state.update(paused=not runtime_state["paused"]),
        on_quit=lambda: runtime_state.update(quit=True),
        is_paused=lambda: runtime_state["paused"],
    )
    if cfg["tray_icon"]["enabled"]:
        tray.start()

    prev_time     = time.time()
    angle         = 0.0
    direction     = "STRAIGHT"
    strength      = 0.0
    throttle_mode = "NEUTRAL"
    throttle_strength = 0.0
    left_open     = False
    right_open    = False
    lost_frames   = 0
    was_paused    = False

    print("=" * 55)
    print("  Gesture Drive  |  Press Q to quit")
    print("=" * 55)
    print("  FIST = Accelerate/Reverse   OPEN = Brake   FISTS TOGETHER = Handbrake")
    print("  PALMS TOGETHER = In-game Pause (Esc)   BOTH THUMBS UP = Confirm (Enter)")
    print("  RIGHT thumb up = Gear up/Drive   LEFT thumb up = Gear down/Reverse")
    print("  Tilt hands = Steer")
    print(f"  [{KB['pause'].upper()}] Pause (works even unfocused, or use the tray icon)")
    print(f"  [{KB['gear_mode_toggle'].upper()}] Toggle manual/automatic gears")
    print("=" * 55)

    show_setup_screen(cap)

    try:
        while True:
            if runtime_state["quit"]:
                break

            if runtime_state["paused"] != was_paused:
                if runtime_state["paused"]:
                    controller.release_all()
                was_paused = runtime_state["paused"]
                tray.refresh()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            apply_trackbars(controller)
            pal = PALETTES[runtime_state["theme"]]

            both_visible = False

            if not runtime_state["paused"]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                if results.multi_hand_landmarks and results.multi_handedness:
                    hand_data = {}

                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=pal["LANDMARK"], thickness=1, circle_radius=2),
                            mp_drawing.DrawingSpec(color=pal["CONNECTION"], thickness=1),
                        )
                        wrist  = hand_landmarks.landmark[0]
                        wx     = int(wrist.x * w)
                        wy     = int(wrist.y * h)
                        opened = is_open_hand(hand_landmarks)
                        hand_data[label] = (wrist.x, wrist.y, wx, wy, opened, hand_landmarks)

                    if "Left" in hand_data and "Right" in hand_data:
                        both_visible = True
                        lost_frames  = 0

                        lx_n, ly_n, lx_px, ly_px, left_open,  left_lm  = hand_data["Left"]
                        rx_n, ry_n, rx_px, ry_px, right_open, right_lm = hand_data["Right"]

                        now = time.time()
                        hand_dist = math.hypot(rx_n - lx_n, ry_n - ly_n)
                        avg_wrist_y = (ly_n + ry_n) / 2.0

                        draw_hand_connection(frame, pal, (lx_px, ly_px), (rx_px, ry_px))
                        angle, direction, strength = controller.update_steer((lx_n, ly_n), (rx_n, ry_n))
                        throttle_mode, throttle_strength = controller.update_throttle(
                            left_open, right_open, avg_wrist_y, hand_dist, now
                        )
                        # Skip gear shifts while the handbrake gesture (fists together) is
                        # active, so bringing fists close together can't be misread as a shift.
                        if not (controller.handbrake_enabled and hand_dist < controller.handbrake_dist):
                            if both_thumbs_up(left_lm, right_lm):
                                controller.update_confirm(now)
                            else:
                                controller.update_gear_shift(left_lm, right_lm, now)
                    else:
                        lost_frames += 1
                        if lost_frames >= cfg["detection"]["grace_frames"]:
                            controller.release_all()
                            angle, direction, strength = 0.0, "STRAIGHT", 0.0
                            throttle_mode, throttle_strength = "NEUTRAL", 0.0
                            left_open = right_open = False
                else:
                    lost_frames += 1
                    if lost_frames >= cfg["detection"]["grace_frames"]:
                        controller.release_all()
                        angle, direction, strength = 0.0, "STRAIGHT", 0.0
                        throttle_mode, throttle_strength = "NEUTRAL", 0.0
                        left_open = right_open = False

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame, pal, angle, direction, strength, throttle_mode, throttle_strength,
                      controller.gear, controller.gear_mode, both_visible, left_open, right_open, fps, cfg["hud"]["show_angle"])

            if runtime_state["show_legend"]:
                draw_legend(frame, pal)

            if runtime_state["paused"]:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "PAUSED", (w // 2 - 100, h // 2), font, 1.4, (0, 180, 255), 3)

            cv2.imshow("Virtual Steering Wheel", frame)

            key = cv2.waitKey(1) & 0xFF
            # Esc is intentionally NOT treated as quit here: it's the default output
            # key for the in-game-pause gesture, sent as a real OS keystroke via pynput.
            # If our own preview window has focus when that gesture fires, cv2.waitKey
            # would see that same Esc and (if it counted as quit) close the app.
            if key in (ord(KB["quit"][0].lower()), ord(KB["quit"][0].upper())):
                runtime_state["quit"] = True
            elif key in (ord(KB["legend"][0].lower()), ord(KB["legend"][0].upper())):
                runtime_state["show_legend"] = not runtime_state["show_legend"]
            elif key in (ord(KB["theme_toggle"][0].lower()), ord(KB["theme_toggle"][0].upper())):
                runtime_state["theme"] = "light" if runtime_state["theme"] == "dark" else "dark"
            elif key in (ord(KB["gear_mode_toggle"][0].lower()), ord(KB["gear_mode_toggle"][0].upper())):
                controller.toggle_gear_mode()

    finally:
        controller.release_all()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        tray.stop()
        hotkey_listener.stop()
        print("\n[INFO] Stopped. All keys released.")


if __name__ == "__main__":
    main()
