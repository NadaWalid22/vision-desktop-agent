"""Vision-Based Desktop Automation with Dynamic Icon Grounding - TJM Labs.

Per post: fresh screenshot -> ground Notepad icon (CLIP-based VisualGrounder)
-> double-click to launch -> type the post into Notepad (visual demonstration)
-> write the post to disk as post_{id}.txt -> close Notepad. The Notepad icon is
re-grounded from a fresh screenshot every iteration (no cached coordinates).
"""
import os, json, time, requests, pyautogui
import numpy as np
import cv2
from src.grounding import VisualGrounder
from src.utils.logger import configure_logging, get_logger

API_URL = "https://jsonplaceholder.typicode.com/posts"
NUM_POSTS = 10
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "tjm-project")
CACHE_FILE = "posts_cache.json"
LOG_FILE = "automation_run.log"
MAX_RETRIES = 3

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.4

configure_logging(level="INFO", log_file=LOG_FILE)
logger = get_logger(__name__)


def fetch_posts(n):
    logger.info("Fetching {} posts from API...", n)
    for attempt in range(1, 4):
        try:
            resp = requests.get(API_URL, timeout=30)
            resp.raise_for_status()
            posts = resp.json()[:n]
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(posts, f)
            logger.info("Fetched and cached {} posts.", len(posts))
            return posts
        except Exception as e:
            logger.warning("API attempt {}/3 failed: {}", attempt, e)
            time.sleep(2)
    if os.path.exists(CACHE_FILE):
        logger.warning("Network unavailable; using cached posts.")
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)[:n]
    raise RuntimeError("No posts available.")


_DARK_THEME_QUERY = (
    "a {name} application icon on a dark Windows desktop, small square icon"
)
_LIGHT_THEME_QUERY = (
    "a {name} application icon on a Windows desktop, small square icon"
)
_DARK_LUMINANCE_THRESHOLD = 85  # mean luminance below this → dark theme


def _capture_bgr() -> np.ndarray:
    """Capture the current screen as a BGR numpy array for VisualGrounder."""
    pil = pyautogui.screenshot()
    rgb = np.asarray(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _is_dark_theme(screenshot_bgr: np.ndarray) -> bool:
    """
    Return True when the desktop is using a dark colour scheme.

    Uses a luminance histogram of the screenshot: convert to greyscale and
    check whether the mean pixel brightness falls below a threshold.
    Dark themes typically have mean luminance well below 128; light themes
    sit above it.  A threshold of 85 gives comfortable headroom between the
    two modes while staying robust to partially-dark wallpapers.
    """
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(np.mean(gray))
    logger.debug("Screen mean luminance: {:.1f}", mean_luminance)
    return mean_luminance < _DARK_LUMINANCE_THRESHOLD


def _wait_for_notepad(timeout: float = 10.0) -> bool:
    """Poll until a window titled 'Notepad' appears, up to *timeout* seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            import pygetwindow as gw
            if gw.getWindowsWithTitle("Notepad"):
                return True
        except ImportError:
            import subprocess
            result = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of every window of every process whose name contains "Notepad"'],
                capture_output=True, text=True, timeout=2,
            )
            if result.stdout.strip():
                return True
        time.sleep(0.5)
    return False


def launch_notepad(grounder: VisualGrounder) -> bool:
    original_threshold = grounder._ranker.confidence_threshold
    for attempt in range(1, MAX_RETRIES + 1):
        screenshot = _capture_bgr()
        dark = _is_dark_theme(screenshot)
        grounder._ranker.dark_mode = dark
        query_template = _DARK_THEME_QUERY if dark else _LIGHT_THEME_QUERY
        logger.info("Theme detected: {}", "dark" if dark else "light")
        candidates = grounder.locate_all(
            screenshot, "notepad", query=query_template.format(name="notepad")
        )
        if candidates:
            if len(candidates) > 1:
                logger.warning(
                    "{} Notepad candidates found — selecting top by CLIP score:", len(candidates)
                )
                for i, c in enumerate(candidates):
                    logger.info(
                        "  [{}] ({}, {}) conf={:.3f} clip={:.3f} ocr='{}'",
                        i + 1, c.x, c.y, c.confidence, c.clip_score, c.detected_text,
                    )
            best = candidates[0]
            logger.info(
                "Selected Notepad at ({}, {}) [conf={:.3f}]", best.x, best.y, best.confidence
            )
            grounder._ranker.confidence_threshold = original_threshold
            pyautogui.moveTo(best.x, best.y, duration=0.3)
            pyautogui.doubleClick()
            if not _wait_for_notepad(timeout=10.0):
                logger.error("Notepad window did not appear within 10s after double-click.")
                return False
            logger.info("Notepad window detected — ready.")
            return True

        grounder._ranker.confidence_threshold = max(
            0.10, original_threshold - 0.05 * attempt
        )
        logger.warning(
            "Attempt {}/{}: no candidates — lowering threshold to {:.2f}",
            attempt, MAX_RETRIES, grounder._ranker.confidence_threshold,
        )

    grounder._ranker.confidence_threshold = original_threshold
    logger.error("Notepad not found after {} attempts.", MAX_RETRIES)
    return False


def type_post(post):
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.3)
    pyautogui.press("delete")
    time.sleep(0.3)
    content = "Title: %s\n\n%s" % (post["title"], post["body"])
    import pyperclip
    pyperclip.copy(content)
    time.sleep(0.15)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.3)


def save_post_to_disk(post):
    content = "Title: %s\n\n%s" % (post["title"], post["body"])
    filepath = os.path.join(SAVE_DIR, "post_%d.txt" % post["id"])
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    exists = os.path.exists(filepath)
    if exists:
        logger.debug("Wrote {}", filepath)
    return exists


def close_notepad():
    pyautogui.hotkey("alt", "F4")
    time.sleep(1.2)
    pyautogui.press("n")   # decline "save changes?" prompt from Notepad
    time.sleep(2)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info("Log file: {}", os.path.abspath(LOG_FILE))

    logger.info("Loading CLIP grounding model (first run may take a moment)...")
    grounder = VisualGrounder()
    logger.info("VisualGrounder ready.")

    posts = fetch_posts(NUM_POSTS)
    logger.info("Got {} posts. Saving to: {}", len(posts), SAVE_DIR)
    logger.info("Starting in 6 seconds — click an empty spot on the desktop now!")
    logger.info("Make sure NO Notepad window is open.")
    time.sleep(6)

    success = 0
    for post in posts:
        logger.info("--- Post {} ---", post["id"])
        try:
            if not launch_notepad(grounder):
                logger.warning("Could not launch Notepad; skipping post {}.", post["id"])
                continue
            type_post(post)
            saved = save_post_to_disk(post)
            close_notepad()
            if saved:
                success += 1
                logger.info("Saved post_{}.txt", post["id"])
            else:
                logger.warning("post_{}.txt was not written.", post["id"])
        except Exception as e:
            logger.exception("ERROR on post {}: {}", post["id"], e)
            try:
                pyautogui.press("esc")
                pyautogui.hotkey("alt", "F4")
                time.sleep(1)
                pyautogui.press("n")
            except Exception:
                pass
            continue

    logger.info("Done. {}/{} posts processed.", success, len(posts))
    logger.info("Files saved in: {}", SAVE_DIR)
    try:
        input("\nPress Enter to close...")
    except Exception:
        pass


if __name__ == "__main__":
    main()
