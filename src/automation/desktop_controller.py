"""
Low-level desktop control: mouse, keyboard, and window management.

Design principles:
- Every action has a configurable delay to avoid race conditions with the OS.
- All methods are idempotent where possible.
- Failures raise DesktopControlError with a descriptive message rather than
  silent swallowing, so the workflow layer can make informed retry decisions.
- Platform detection allows graceful degradation: the core mouse/keyboard
  actions work cross-platform via pyautogui; Windows-specific window management
  (pygetwindow) is loaded lazily and silently bypassed on Linux/macOS.
"""

from __future__ import annotations

import platform
import sys
import time
from contextlib import contextmanager
from typing import Generator

import pyautogui  # type: ignore[import]

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Safety: fail-safe corner — moving mouse to (0, 0) aborts pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # default inter-action pause


class DesktopControlError(RuntimeError):
    """Raised when a desktop control action fails irrecoverably."""


class DesktopController:
    """
    Abstracts mouse, keyboard, and window focus operations.

    Args:
        action_delay: Extra seconds to sleep after each action (allows
                      the OS UI to catch up on slower machines).
        failsafe:     If True, moving mouse to (0,0) raises an abort exception.
    """

    def __init__(
        self,
        action_delay: float = 0.3,
        failsafe: bool = True,
    ) -> None:
        self._delay = action_delay
        pyautogui.FAILSAFE = failsafe
        self._is_windows = sys.platform == "win32"
        self._is_mac = sys.platform == "darwin"

    # ─── Mouse actions ────────────────────────────────────────────────────────

    def move_to(self, x: int, y: int, duration: float = 0.25) -> None:
        """Smoothly move cursor to (x, y)."""
        logger.debug(f"Move to ({x}, {y})")
        pyautogui.moveTo(x, y, duration=duration)
        self._sleep()

    def click(self, x: int, y: int, button: str = "left") -> None:
        """Single click at (x, y)."""
        logger.debug(f"Click ({x}, {y}) [{button}]")
        pyautogui.click(x, y, button=button)
        self._sleep()

    def double_click(self, x: int, y: int) -> None:
        """
        Double-click at (x, y).

        Used to open icons from the desktop or file manager.
        We move first so the user can see the cursor, then double-click
        after a brief pause to avoid OS-level misfire detection.
        """
        logger.info(f"Double-click at ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.3)
        time.sleep(0.1)
        pyautogui.doubleClick(x, y)
        self._sleep(extra=0.5)

    def right_click(self, x: int, y: int) -> None:
        """Right-click context menu at (x, y)."""
        pyautogui.rightClick(x, y)
        self._sleep()

    # ─── Keyboard actions ─────────────────────────────────────────────────────

    def type_text(self, text: str, interval: float = 0.02) -> None:
        """
        Type a string into the currently focused element.

        Uses pyautogui.typewrite for ASCII-safe strings.  For strings with
        special characters or Unicode, falls back to clipboard paste.
        """
        logger.debug(f"Type text: {text[:40]}{'…' if len(text) > 40 else ''}")
        if self._text_is_ascii_safe(text):
            pyautogui.typewrite(text, interval=interval)
        else:
            self._paste_text(text)
        self._sleep()

    def press_key(self, *keys: str) -> None:
        """
        Press one or more keys.  If multiple keys provided, they are pressed
        simultaneously as a hotkey (e.g. press_key('ctrl', 's')).
        """
        logger.debug(f"Press key(s): {keys}")
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        self._sleep()

    def hotkey(self, *keys: str) -> None:
        """Alias for pressing a key combination."""
        self.press_key(*keys)

    def type_line(self, text: str) -> None:
        """Type text followed by Enter."""
        self.type_text(text)
        self.press_key("enter")

    # ─── Window management ────────────────────────────────────────────────────

    def focus_window(self, title_substring: str, timeout: float = 10.0) -> bool:
        """
        Bring a window with *title_substring* in its title to the foreground.

        Returns True if the window was found and focused, False on timeout.
        """
        logger.debug(f"Focusing window containing: '{title_substring}'")
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                win = self._find_window(title_substring)
                if win is not None:
                    self._activate_window(win)
                    time.sleep(0.4)
                    return True
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Window focus attempt failed: {e}")
            time.sleep(0.5)

        logger.warning(f"Window '{title_substring}' not found within {timeout}s")
        return False

    def close_window(self, title_substring: str) -> bool:
        """Close a window by title substring. Returns True if window found."""
        try:
            win = self._find_window(title_substring)
            if win is not None:
                self._activate_window(win)
                time.sleep(0.3)
                self.press_key("alt", "f4")
                time.sleep(0.5)
                return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Close window failed: {e}")
        return False

    def is_window_open(self, title_substring: str) -> bool:
        """Return True if any window title contains *title_substring*."""
        return self._find_window(title_substring) is not None

    def dismiss_dialog(self) -> None:
        """
        Attempt to dismiss unexpected modal dialogs (Save?, Error, etc.)
        by pressing Escape then Enter.  Used as a recovery action.
        """
        logger.info("Attempting to dismiss unexpected dialog")
        time.sleep(0.3)
        pyautogui.press("escape")
        time.sleep(0.2)
        pyautogui.press("enter")
        time.sleep(0.3)

    # ─── Save / file helpers ──────────────────────────────────────────────────

    def save_as(self, filepath: str) -> None:
        """
        Trigger Save As dialog (Ctrl+Shift+S or Ctrl+S depending on OS),
        type the file path, and confirm.
        """
        logger.info(f"Save As: {filepath}")
        self.hotkey("ctrl", "shift", "s")
        time.sleep(1.0)  # Wait for dialog

        # Type the full path into the filename field
        self._paste_text(filepath)
        time.sleep(0.3)
        self.press_key("enter")
        time.sleep(0.5)

        # Handle "replace file?" confirmation dialog if it appears
        self.press_key("enter")
        time.sleep(0.3)

    def ctrl_save(self) -> None:
        """Save with Ctrl+S."""
        logger.info("Save (Ctrl+S)")
        self.hotkey("ctrl", "s")
        self._sleep(extra=0.5)

    # ─── Screen info ──────────────────────────────────────────────────────────

    @staticmethod
    def screen_size() -> tuple[int, int]:
        """Return (width, height) of the primary monitor."""
        return pyautogui.size()

    # ─── Context manager: safe action block ───────────────────────────────────

    @contextmanager
    def safe_action(self, description: str) -> Generator:
        """
        Context manager that logs action start/end and catches/re-raises
        exceptions with added context.

        Usage::

            with controller.safe_action("open Notepad"):
                controller.double_click(x, y)
        """
        logger.debug(f"[safe_action] Starting: {description}")
        try:
            yield
            logger.debug(f"[safe_action] Done: {description}")
        except pyautogui.FailSafeException:
            raise DesktopControlError(
                "PyAutoGUI fail-safe triggered (mouse at corner). "
                "Move mouse away from screen corners."
            )
        except Exception as e:
            raise DesktopControlError(
                f"Action '{description}' failed: {type(e).__name__}: {e}"
            ) from e

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _sleep(self, extra: float = 0.0) -> None:
        time.sleep(self._delay + extra)

    def _find_window(self, title_substring: str):
        """Return a window object matching the title substring, or None."""
        if self._is_windows:
            try:
                import pygetwindow as gw  # type: ignore[import]
                wins = gw.getWindowsWithTitle(title_substring)
                return wins[0] if wins else None
            except ImportError:
                pass

        if self._is_mac:
            try:
                import subprocess
                result = subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to get name of every window of every process whose name contains "{title_substring}"'],
                    capture_output=True, text=True, timeout=2
                )
                return result.stdout.strip() or None
            except Exception:
                pass

        return None

    def _activate_window(self, win) -> None:
        """Bring window to foreground. Handles both pygetwindow and macOS."""
        if hasattr(win, "activate"):
            try:
                win.restore()
            except Exception:
                pass
            win.activate()
        elif hasattr(win, "focus"):
            win.focus()

    @staticmethod
    def _text_is_ascii_safe(text: str) -> bool:
        """Check if text can be typed safely with pyautogui.typewrite."""
        try:
            text.encode("ascii")
            # pyautogui typewrite also struggles with newlines > len 1
            return "\n" not in text and len(text) < 500
        except UnicodeEncodeError:
            return False

    def _paste_text(self, text: str) -> None:
        """
        Copy *text* to clipboard then paste.  Handles Unicode and long strings.
        """
        import pyperclip  # type: ignore[import]
        pyperclip.copy(text)
        time.sleep(0.1)
        self.hotkey("ctrl", "v")
