"""Vision-Based Desktop Automation with Dynamic Icon Grounding - TJM Labs.

Per post: fresh screenshot -> ground Notepad icon (OmniParser visual grounding)
-> double-click to launch -> type the post into Notepad (visual demonstration)
-> write the post to disk as post_{id}.txt -> close Notepad. The Notepad icon is
re-grounded from a fresh screenshot every iteration (no cached coordinates).
"""
import os, json, time, requests, pyautogui
import numpy as np
from PIL import Image
from util.utils import (
    get_yolo_model,
    get_caption_model_processor,
    check_ocr_box,
    predict_yolo,
    get_parsed_content_icon,
)

API_URL = "https://jsonplaceholder.typicode.com/posts"
NUM_POSTS = 10
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "tjm-project")
SCREENSHOT_PATH = "current_screen.png"
CACHE_FILE = "posts_cache.json"
TARGET_LABEL = "notepad"
MAX_RETRIES = 3

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.4


def fetch_posts(n):
    print("Fetching %d posts from API..." % n)
    for attempt in range(1, 4):
        try:
            resp = requests.get(API_URL, timeout=30)
            resp.raise_for_status()
            posts = resp.json()[:n]
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(posts, f)
            print("  Fetched and cached %d posts." % len(posts))
            return posts
        except Exception as e:
            print("  API attempt %d/3 failed: %s" % (attempt, e))
            time.sleep(2)
    if os.path.exists(CACHE_FILE):
        print("  Network unavailable; using cached posts.")
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)[:n]
    raise RuntimeError("No posts available.")


def ground_notepad(yolo_model, caption_processor):
    """
    Full OmniParser grounding pipeline:
      1. Screenshot -> EasyOCR (fast text match first)
      2. YOLO icon detection -> Florence-2 captions -> semantic match
    Returns (cx, cy) pixel coords of the Notepad icon, or None.
    """
    pyautogui.hotkey("win", "d")
    time.sleep(0.8)
    pyautogui.screenshot().save(SCREENSHOT_PATH)
    image = Image.open(SCREENSHOT_PATH).convert("RGB")
    w, h = image.size
    image_np = np.asarray(image)

    # --- Stage 1: fast OCR pass (finds taskbar / desktop text labels) ---
    ocr_rslt, _ = check_ocr_box(
        SCREENSHOT_PATH, display_img=False, output_bb_format="xyxy", use_paddleocr=False
    )
    ocr_text, ocr_bbox = ocr_rslt
    for t, box in zip(ocr_text, ocr_bbox):
        if TARGET_LABEL in t.lower():
            x1, y1, x2, y2 = box
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    # --- Stage 2: YOLO icon detection + Florence-2 captions ---
    import torch
    boxes_xyxy, _conf, _phrases = predict_yolo(
        model=yolo_model, image=image, box_threshold=0.05, imgsz=(h, w), scale_img=False
    )
    if len(boxes_xyxy) == 0:
        return None

    # Normalise to [0,1] for get_parsed_content_icon
    boxes_norm = boxes_xyxy / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes_xyxy.device)

    captions = get_parsed_content_icon(
        filtered_boxes=boxes_norm,
        starting_idx=0,
        image_source=image_np,
        caption_model_processor=caption_processor,
    )

    for caption, box in zip(captions, boxes_xyxy.tolist()):
        if TARGET_LABEL in caption.lower():
            x1, y1, x2, y2 = box
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    return None


def launch_notepad(yolo_model, caption_processor):
    for attempt in range(1, MAX_RETRIES + 1):
        coords = ground_notepad(yolo_model, caption_processor)
        if coords:
            x, y = coords
            print("  Grounded Notepad at (%d, %d) [attempt %d]" % (x, y, attempt))
            pyautogui.moveTo(x, y, duration=0.3)
            pyautogui.doubleClick()
            time.sleep(3)
            return True
        print("  Notepad not found (attempt %d/%d), retrying..." % (attempt, MAX_RETRIES))
        time.sleep(1)
    return False


def type_post(post):
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.3)
    pyautogui.press("delete")
    time.sleep(0.3)
    content = "Title: %s\n\n%s" % (post["title"], post["body"])
    pyautogui.typewrite(content, interval=0.01)
    time.sleep(0.4)


def save_post_to_disk(post):
    content = "Title: %s\n\n%s" % (post["title"], post["body"])
    filepath = os.path.join(SAVE_DIR, "post_%d.txt" % post["id"])
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.exists(filepath)


def close_notepad():
    pyautogui.hotkey("alt", "F4")
    time.sleep(1.2)
    pyautogui.press("n")   # decline "save changes?" prompt from Notepad
    time.sleep(2)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading grounding models (CPU, this takes a moment)...")
    yolo_model = get_yolo_model("weights/icon_detect/model.pt")
    caption_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="weights/icon_caption_florence",
    )
    print("Models loaded.\n")

    posts = fetch_posts(NUM_POSTS)
    print("Got %d posts. Saving to: %s\n" % (len(posts), SAVE_DIR))
    print("Starting in 6 seconds - click an empty spot on the desktop now!")
    print("Make sure NO Notepad window is open.")
    time.sleep(6)

    success = 0
    for post in posts:
        print("\n--- Post %d ---" % post["id"])
        try:
            if not launch_notepad(yolo_model, caption_processor):
                print("  Could not launch Notepad; skipping.")
                continue
            type_post(post)
            saved = save_post_to_disk(post)
            close_notepad()
            if saved:
                success += 1
                print("  Saved post_%d.txt" % post["id"])
            else:
                print("  WARNING: post_%d.txt not written." % post["id"])
        except Exception as e:
            print("  ERROR on post %d: %s" % (post["id"], e))
            try:
                pyautogui.press("esc")
                pyautogui.hotkey("alt", "F4")
                time.sleep(1)
                pyautogui.press("n")
            except Exception:
                pass
            continue

    print("\nDone. %d/%d posts processed." % (success, len(posts)))
    print("Files saved in: %s" % SAVE_DIR)
    try:
        input("\nPress Enter to close...")
    except Exception:
        pass


if __name__ == "__main__":
    main()
