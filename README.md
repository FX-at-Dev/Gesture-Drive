# Gesture Drive

Control a driving game with hand gestures. A webcam and [MediaPipe](https://developers.google.com/mediapipe) hand tracking read your hand positions and translate them into keyboard input (`W`/`A`/`S`/`D` by default) via `pynput`.

## Requirements

| Package        | Version   |
|----------------|-----------|
| Python         | >= 3.9    |
| mediapipe      | >= 0.10.0 |
| opencv-python  | >= 4.8.0  |
| pynput         | >= 1.7.6  |
| numpy          | >= 1.24.0 |
| pystray        | >= 0.19.0 |
| pillow         | >= 10.0.0 |

`pystray` and `pillow` are only needed for the system tray icon — the app still runs without them, just with the tray icon disabled.

A webcam is required. On macOS, grant camera access under System Settings > Privacy & Security > Camera.

## Install

```bash
# create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

## Run

```bash
python steering_wheel.py
```

A setup screen with a short countdown appears first; press any key to skip it. Press `Q` to quit.

## Gestures

| Gesture                                      | Action                          |
|-----------------------------------------------|----------------------------------|
| Tilt both hands left / right (like a wheel)   | Steer left / right               |
| Make a fist with **both** hands                | Accelerate (`W`) — or Reverse (`S`) if the current gear is `R` — strength scales with hand height |
| Open **both** palms flat                       | Brake (`S`) — strength scales with hand height |
| Make fists with both hands **and** bring them close together | Handbrake (`Space`) |
| Open **both** palms **and** bring them close together | In-game pause (`Esc`, tapped once per gesture) |
| Thumbs up with **both** hands at once | Confirm (`Enter`, tapped once per gesture) — for menu selections |
| **Right hand** thumbs up (other fingers curled, thumb clearly extended) | Shift up: gear +1 in Manual, selects `D` in Automatic |
| **Left hand** thumbs up (other fingers curled, thumb clearly extended) | Shift down: gear −1 in Manual, selects `R` in Automatic |
| One hand open, one hand fisted                 | Neutral — no throttle/brake |
| Both hands out of frame for ~8 frames          | All keys released automatically |

Both shifts use the same thumbs-up gesture — which hand does it decides the direction, so there's no need for an awkward thumbs-down. When **both** hands go thumbs-up at once it's read as Confirm instead, so it never also triggers a gear shift. Gear shifts and Confirm are both ignored while the handbrake gesture is active. Raising both hands higher in the frame increases accelerate/brake/reverse intensity; lowering them reduces it. Both hands must be visible for steering, throttle, gear-shift, and confirm gestures to register.

The in-game pause and confirm gestures are distinct from the `P` key below: they send a single `Esc`/`Enter` tap to whatever game/app has focus — for that game's own pause/menu handling — instead of pausing Gesture Drive's own gesture tracking. Both are debounced (1.2s by default) so holding the pose doesn't spam the key.

Gears run `R, 1, 2, ... 6`. In **Manual** mode each shift moves one gear at a time. In **Automatic** mode there's no numbered gears — right-hand thumbs up always selects `D` (drive) and left-hand thumbs up always selects `R` (reverse).

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `P` | Pause / resume gesture control — works even if the app window isn't focused, and is also available from the system tray icon |
| `G` | Toggle Manual / Automatic gear mode |
| `H` | Toggle the on-screen gesture/keys legend |
| `T` | Toggle light / dark theme |

`Esc` is deliberately **not** a quit shortcut: it's the output key the in-game-pause gesture sends, and treating incoming Esc as "quit" would close Gesture Drive itself whenever that gesture fired while the preview window had focus.

All of the above key bindings, plus the output keys (`W`/`A`/`S`/`D`/`Space`/`Esc`/`Enter`/`Shift`/`Ctrl`, including a separate `reverse` binding if your game expects a different key than brake), dead zones, smoothing, throttle sensitivity, handbrake distance, gear-shift timing, and the pause/confirm gestures' distance/debounce are all configurable in `config.json` (auto-created on first run) — no code changes needed. A "Controls" window with live sliders also lets you tune dead zone, release zone, soft zone, smoothing, and handbrake distance while the app is running.

## System tray

When `pystray` and `pillow` are installed, a tray icon appears with Pause/Resume and Quit options, so the app can run in the background without keeping the camera window focused.
