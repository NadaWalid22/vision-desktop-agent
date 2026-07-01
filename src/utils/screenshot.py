"""
Screenshot capture utilities.

Uses mss (multi-screen shot) as the primary backend because it is
significantly faster than PIL.ImageGrab or pyautogui.screenshot for
repeated captures in a hot loop — it accesses the framebuffer directly
without going through the OS clipboard.

The module returns BGR numpy arrays (OpenCV convention) so they can be
fed directly into the OpenCV-based grounding pipeline without conversion.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2  # type: ignore[import]
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScreenshotCapture:
    """
    Fast screen capture using mss.

    Args:
        monitor: Monitor index (1 = primary). 0 = all monitors combined.
        save_dir: If set, captured screenshots are also saved here.
    """

    def __init__(
        self,
        monitor: int = 1,
        save_dir: Path | str | None = None,
    ) -> None:
        self._monitor_idx = monitor
        self._save_dir = Path(save_dir) if save_dir else None

    def capture(self) -> NDArray:
        """
        Capture the full desktop screenshot.

        Returns:
            BGR uint8 array of shape (H, W, 3).

        Performance: ~15–30 ms per call on a 1920×1080 display (mss).
        """
        try:
            import mss  # type: ignore[import]
            import mss.tools

            with mss.mss() as sct:
                monitor = sct.monitors[self._monitor_idx]
                sct_img = sct.grab(monitor)
                # mss returns BGRA; drop alpha channel
                arr = np.array(sct_img)[:, :, :3]  # (H, W, 3) BGR
                return arr

        except Exception as e:
            logger.warning(f"mss capture failed ({e}) — falling back to pyautogui")
            return self._capture_pyautogui()

    def capture_region(self, x: int, y: int, w: int, h: int) -> NDArray:
        """
        Capture a sub-region of the screen.

        Args:
            x, y: Top-left corner in screen coordinates.
            w, h: Width and height in pixels.

        Returns:
            BGR uint8 array of shape (h, w, 3).
        """
        try:
            import mss

            with mss.mss() as sct:
                region = {"left": x, "top": y, "width": w, "height": h}
                sct_img = sct.grab(region)
                return np.array(sct_img)[:, :, :3]
        except Exception:
            full = self.capture()
            return full[y : y + h, x : x + w]

    def save(
        self,
        image: NDArray,
        path: Path | str,
        quality: int = 95,
    ) -> Path:
        """
        Save a BGR numpy array as PNG (or JPEG if path ends in .jpg/.jpeg).

        Returns the saved path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        logger.debug(f"Screenshot saved: {path}")
        return path

    def capture_and_save(
        self, path: Path | str | None = None, label: str = ""
    ) -> tuple[NDArray, Path | None]:
        """
        Convenience: capture + optionally save.  Returns (array, saved_path).
        """
        arr = self.capture()
        saved: Path | None = None

        if path:
            saved = self.save(arr, path)
        elif self._save_dir:
            ts = int(time.time() * 1000)
            filename = f"screenshot_{ts}{'_' + label if label else ''}.png"
            saved = self.save(arr, self._save_dir / filename)

        return arr, saved

    # ─── Annotation helpers ───────────────────────────────────────────────────

    @staticmethod
    def annotate(
        image: NDArray,
        x: int,
        y: int,
        w: int,
        h: int,
        confidence: float,
        label: str = "",
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> NDArray:
        """
        Draw a detection bounding box and confidence label on a copy of *image*.

        Returns the annotated copy (original is unchanged).
        """
        out = image.copy()

        # Bounding box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

        # Centre crosshair
        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)

        # Label background + text
        text = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, tf = 0.6, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, tf)
        bg_y1 = max(0, y - th - baseline - 4)
        cv2.rectangle(out, (x, bg_y1), (x + tw + 4, y), color, -1)
        cv2.putText(
            out, text,
            (x + 2, y - baseline - 2),
            font, scale,
            (0, 0, 0),  # black text on coloured background
            tf, cv2.LINE_AA,
        )

        return out

    # ─── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _capture_pyautogui() -> NDArray:
        """Fallback screenshot via pyautogui (slower but universally available)."""
        import pyautogui  # type: ignore[import]
        from PIL import Image

        pil = pyautogui.screenshot()
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
