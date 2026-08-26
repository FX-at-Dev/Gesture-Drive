"""Optional system tray icon for Gesture Drive.

Runs in a background thread and lets the app be paused/resumed or quit
from the tray without needing focus on the OpenCV window. Degrades to a
no-op (with a console note) if pystray/Pillow aren't installed.
"""

import threading

try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False


def _make_icon_image(paused):
    size = 64
    color = (255, 170, 0) if paused else (60, 200, 120)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((4, 4, size - 4, size - 4), fill=color)
    draw.ellipse((22, 22, 42, 42), fill=(20, 20, 20))
    return img


class TrayIcon:
    """Wraps a pystray.Icon; no-op when pystray/Pillow are unavailable."""

    def __init__(self, on_toggle_pause, on_quit, is_paused):
        self.available = TRAY_AVAILABLE
        self._on_toggle_pause = on_toggle_pause
        self._on_quit = on_quit
        self._is_paused = is_paused
        self._icon = None
        self._thread = None

        if not self.available:
            print("[INFO] pystray/Pillow not installed - system tray icon disabled.")
            print("       Install with: pip install pystray pillow")

    def _build_icon(self):
        menu = pystray.Menu(
            pystray.MenuItem(self._pause_label, self._handle_toggle_pause),
            pystray.MenuItem("Quit", self._handle_quit),
        )
        return pystray.Icon(
            "gesture_drive",
            _make_icon_image(self._is_paused()),
            "Gesture Drive",
            menu,
        )

    def _pause_label(self, item):
        return "Resume" if self._is_paused() else "Pause"

    def _handle_toggle_pause(self, icon, item):
        self._on_toggle_pause()
        icon.icon = _make_icon_image(self._is_paused())

    def _handle_quit(self, icon, item):
        self._on_quit()
        icon.stop()

    def start(self):
        if not self.available:
            return
        self._icon = self._build_icon()
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def refresh(self):
        """Call after external pause-state changes (e.g. a hotkey) to sync the icon."""
        if self.available and self._icon:
            self._icon.icon = _make_icon_image(self._is_paused())

    def stop(self):
        if self.available and self._icon:
            self._icon.stop()
