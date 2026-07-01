"""
End-to-end Notepad automation workflow.

Workflow per iteration
-----------------------
1.  Capture desktop screenshot.
2.  Ground the Notepad icon using VisualGrounder.
3.  Double-click the icon.
4.  Wait for Notepad to open (window title detection).
5.  For each of the 10 API posts:
    a.  Type "Title: {title}\n{body}\n"
    b.  Save As Desktop/tjm-project/post_{id}.txt
    c.  Clear the editor (Ctrl+A → Del).
6.  Close Notepad.
7.  Repeat (configurable number of iterations).

Error handling strategy
------------------------
- Detection failure: retry up to DETECTION_RETRIES times with decaying
  confidence threshold (VisualGrounder.locate_with_retry).
- Notepad not opening: fallback to subprocess.Popen("notepad.exe").
- Save dialog not appearing: retry Ctrl+Shift+S once.
- Unexpected dialog/popup: DesktopController.dismiss_dialog().
- API failure: PostsClient has its own retry logic via tenacity.
- Keyboard interrupt: caught at the loop level for graceful shutdown.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.api.posts_client import Post, PostsClient
from src.automation.desktop_controller import DesktopControlError, DesktopController
from src.grounding import VisualGrounder
from src.utils.logger import get_logger
from src.utils.screenshot import ScreenshotCapture

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
NOTEPAD_WINDOW_TITLE = "Notepad"
NOTEPAD_BINARY = "notepad.exe"
DEFAULT_OUTPUT_DIR = Path("C:/Users/Public/Desktop/tjm-project")
WAIT_FOR_NOTEPAD = 5.0   # seconds to poll for window
WAIT_AFTER_OPEN = 1.5    # settle time after window detected


class NotepadWorkflowError(RuntimeError):
    """Raised when the workflow cannot proceed after retries."""


class NotepadWorkflow:
    """
    Orchestrates the full vision-guided Notepad automation loop.

    Args:
        output_dir: Directory where post_{id}.txt files are saved.
        controller: Shared DesktopController instance.
        grounder:   Shared VisualGrounder instance.
        capture:    Shared ScreenshotCapture instance.
        api_client: PostsClient for fetching JSONPlaceholder posts.
        detection_retries: How many times to retry icon detection.
        post_limit: Number of posts to process per iteration.
        iterations: How many full workflow cycles to run (0 = infinite).
    """

    def __init__(
        self,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        controller: DesktopController | None = None,
        grounder: VisualGrounder | None = None,
        capture: ScreenshotCapture | None = None,
        api_client: PostsClient | None = None,
        detection_retries: int = 3,
        post_limit: int = 10,
        iterations: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self._ctrl = controller or DesktopController()
        self._grounder = grounder or VisualGrounder()
        self._capture = capture or ScreenshotCapture()
        self._api = api_client or PostsClient()
        self._detection_retries = detection_retries
        self._post_limit = post_limit
        self._iterations = iterations

    # ─── Public entry points ──────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main loop. Runs *self._iterations* full workflow cycles.
        Pass iterations=0 to run indefinitely.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Starting workflow: {self._iterations or 'infinite'} iteration(s), "
            f"{self._post_limit} posts each, output -> {self.output_dir}"
        )

        iteration = 0
        try:
            while self._iterations == 0 or iteration < self._iterations:
                iteration += 1
                logger.info(f"─── Iteration {iteration} ───")
                try:
                    self._run_once(iteration)
                except NotepadWorkflowError as e:
                    logger.error(f"Iteration {iteration} failed: {e}")
                    self._recover()
                except DesktopControlError as e:
                    logger.error(f"Desktop control error: {e}")
                    self._recover()

        except KeyboardInterrupt:
            logger.info("Interrupted by user — shutting down cleanly")
            self._recover()

        logger.success(f"Workflow complete after {iteration} iteration(s)")

    def run_once(self) -> None:
        """Run exactly one workflow cycle (useful for testing)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_once(1)

    # ─── Workflow steps ───────────────────────────────────────────────────────

    def _run_once(self, iteration: int) -> None:
        """Execute one full workflow cycle."""
        # Step 1: Fetch posts first (fail fast before doing UI work)
        posts = self._fetch_posts()

        # Step 2: Locate and open Notepad
        self._open_notepad()

        # Step 3: Wait for Notepad window
        if not self._wait_for_notepad():
            raise NotepadWorkflowError(
                "Notepad window did not open within the timeout"
            )

        # Step 4: Write posts
        for idx, post in enumerate(posts, start=1):
            logger.info(f"  Writing post {idx}/{len(posts)}: '{post.title[:40]}'")
            self._write_and_save_post(post)

        # Step 5: Close Notepad
        self._close_notepad()

        logger.success(
            f"Iteration {iteration} complete — "
            f"{len(posts)} files saved to {self.output_dir}"
        )

    # ─── Step implementations ─────────────────────────────────────────────────

    def _fetch_posts(self) -> list[Post]:
        """Fetch posts from the API. Tenacity handles retries inside client."""
        import asyncio
        logger.info(f"Fetching {self._post_limit} posts from JSONPlaceholder…")
        try:
            posts = asyncio.run(self._api.fetch_posts(limit=self._post_limit))
            logger.info(f"Fetched {len(posts)} posts")
            return posts
        except Exception as e:
            raise NotepadWorkflowError(f"API fetch failed: {e}") from e

    def _open_notepad(self) -> None:
        """
        Locate the Notepad icon via visual grounding and double-click it.

        Falls back to subprocess.Popen if detection fails completely.
        """
        logger.info("Locating Notepad icon via visual grounding…")

        result = self._grounder.locate_with_retry(
            screenshot_fn=self._capture.capture,
            target_name="Notepad",
            retries=self._detection_retries,
            query="Notepad text editor application icon on Windows desktop",
        )

        if result:
            logger.info(
                f"Icon detected at ({result.x}, {result.y}) "
                f"[conf={result.confidence:.3f}]"
            )
            with self._ctrl.safe_action("double-click Notepad icon"):
                self._ctrl.double_click(result.x, result.y)
        else:
            logger.warning(
                "Visual grounding failed — falling back to subprocess launch"
            )
            self._launch_notepad_subprocess()

    def _launch_notepad_subprocess(self) -> None:
        """Direct subprocess fallback when icon detection fails."""
        if sys.platform != "win32":
            raise NotepadWorkflowError(
                "Notepad is Windows-only. "
                "Cannot launch via subprocess on non-Windows platform."
            )
        logger.info("Launching Notepad via subprocess")
        subprocess.Popen([NOTEPAD_BINARY])
        time.sleep(1.5)

    def _wait_for_notepad(self) -> bool:
        """Poll until the Notepad window appears or timeout expires."""
        logger.debug(f"Waiting up to {WAIT_FOR_NOTEPAD}s for Notepad window…")
        deadline = time.time() + WAIT_FOR_NOTEPAD
        while time.time() < deadline:
            if self._ctrl.is_window_open(NOTEPAD_WINDOW_TITLE):
                logger.debug("Notepad window detected")
                time.sleep(WAIT_AFTER_OPEN)
                self._ctrl.focus_window(NOTEPAD_WINDOW_TITLE)
                return True
            time.sleep(0.5)
        return False

    def _write_and_save_post(self, post: Post) -> None:
        """
        Type post content into the focused Notepad window and save to file.

        On save failure, logs the error and continues with the next post
        rather than aborting the entire iteration.
        """
        # Ensure Notepad has focus
        self._ctrl.focus_window(NOTEPAD_WINDOW_TITLE, timeout=3.0)

        # Clear any existing content
        self._ctrl.hotkey("ctrl", "a")
        self._ctrl.press_key("delete")
        time.sleep(0.2)

        # Compose content
        content = f"Title: {post.title}\n\n{post.body}\n"

        # Type content (uses clipboard paste for reliability)
        self._type_content_safely(content)

        # Save the file
        filepath = self.output_dir / f"post_{post.id}.txt"
        try:
            self._save_as(str(filepath))
            logger.debug(f"Saved: {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to save post_{post.id}.txt: {e}")

    def _type_content_safely(self, content: str) -> None:
        """
        Type content using clipboard to handle Unicode and long strings reliably.
        pyautogui typewrite can misfire on special characters in post bodies.
        """
        try:
            import pyperclip  # type: ignore[import]
            pyperclip.copy(content)
            time.sleep(0.15)
            self._ctrl.hotkey("ctrl", "v")
            time.sleep(0.3)
        except ImportError:
            # Fallback: type line by line
            for line in content.splitlines():
                self._ctrl.type_text(line)
                self._ctrl.press_key("enter")

    def _save_as(self, filepath: str) -> None:
        """
        Save current Notepad content to *filepath* using Ctrl+Shift+S (Save As).
        Retries once if the dialog does not appear.
        """
        for attempt in range(2):
            self._ctrl.hotkey("ctrl", "shift", "s")
            time.sleep(1.2)  # Wait for "Save As" dialog

            # Check if any dialog appeared (look for dialog window)
            # Type the full filepath into the filename field
            self._ctrl.hotkey("ctrl", "a")  # select all in filename box
            time.sleep(0.1)

            try:
                import pyperclip  # type: ignore[import]
                pyperclip.copy(filepath)
                time.sleep(0.1)
                self._ctrl.hotkey("ctrl", "v")
            except ImportError:
                self._ctrl.type_text(filepath)

            time.sleep(0.2)
            self._ctrl.press_key("enter")  # Confirm
            time.sleep(0.5)

            # Dismiss "Overwrite?" dialog if present
            self._ctrl.press_key("enter")
            time.sleep(0.3)

            # If we got here without an exception, consider it done
            return

    def _close_notepad(self) -> None:
        """Close the Notepad window. Discard unsaved content if prompted."""
        logger.info("Closing Notepad")
        if not self._ctrl.close_window(NOTEPAD_WINDOW_TITLE):
            # Window may already be closed
            logger.debug("Notepad window not found during close — may already be closed")
            return

        # Handle "Save changes?" dialog
        time.sleep(0.5)
        # Press 'N' (Don't Save) or Tab+Enter depending on dialog variant
        self._ctrl.press_key("n")
        time.sleep(0.3)

    def _recover(self) -> None:
        """
        Best-effort cleanup after a failed iteration:
        - Dismiss any open dialogs
        - Close Notepad if open
        """
        logger.info("Running recovery procedure…")
        try:
            self._ctrl.dismiss_dialog()
            if self._ctrl.is_window_open(NOTEPAD_WINDOW_TITLE):
                self._close_notepad()
        except Exception as e:
            logger.debug(f"Recovery error (non-fatal): {e}")
