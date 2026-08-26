"""Configuration loading, defaults, and key-name resolution for Gesture Drive."""

import json
import os
from pynput.keyboard import Key

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG = {
    "camera_index": 1,
    "flip_camera": True,

    "steering": {
        "dead_zone_deg": 12,
        "release_zone_deg": 6,
        "soft_zone_deg": 25,
        "smoothing_alpha": 0.35,
    },

    "throttle": {
        "pwm_period_ms": 150,
        "min_intensity_y": 0.15,
        "max_intensity_y": 0.85,
    },

    "handbrake": {
        "enabled": True,
        "distance_thresh": 0.12,
    },

    "game_pause": {
        "enabled": True,
        "distance_thresh": 0.12,
        "debounce_ms": 1200,
    },

    "confirm": {
        "enabled": True,
        "debounce_ms": 1200,
    },

    "gear_shift": {
        "enabled": True,
        "debounce_ms": 900,
        "min_gear": 0,
        "max_gear": 6,
        "start_mode": "manual",
    },

    "detection": {
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
        "grace_frames": 8,
        "open_finger_thresh": 3,
    },

    "keybindings": {
        "steer_left": "a",
        "steer_right": "d",
        "accel": "w",
        "brake": "s",
        "reverse": "s",
        "handbrake": "space",
        "gear_up": "shift",
        "gear_down": "ctrl",
        "game_pause": "esc",
        "confirm": "enter",
        "pause": "p",
        "legend": "h",
        "theme_toggle": "t",
        "gear_mode_toggle": "g",
        "quit": "q",
    },

    "theme": "dark",

    "hud": {
        "show_angle": True,
    },

    "tray_icon": {
        "enabled": True,
    },
}

# Names usable in config.json for non-character keys, resolved to pynput Key members.
SPECIAL_KEYS = {
    "space": Key.space,
    "shift": Key.shift,
    "ctrl": Key.ctrl,
    "alt": Key.alt,
    "enter": Key.enter,
    "esc": Key.esc,
    "tab": Key.tab,
    "backspace": Key.backspace,
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
}


def resolve_key(name):
    """Turn a config key-name string into a pynput key (Key enum or single char)."""
    name = str(name).strip().lower()
    if name in SPECIAL_KEYS:
        return SPECIAL_KEYS[name]
    return name[0] if name else name


def _deep_merge(defaults, overrides):
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path=CONFIG_PATH):
    """Load config.json, merging it onto defaults. Creates the file if missing."""
    if not os.path.exists(path):
        save_config(DEFAULT_CONFIG, path)
        return json.loads(json.dumps(DEFAULT_CONFIG))

    try:
        with open(path, "r") as f:
            user_cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read {path} ({e}); using defaults.")
        return json.loads(json.dumps(DEFAULT_CONFIG))

    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
