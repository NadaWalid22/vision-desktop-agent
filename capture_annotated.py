"""Capture an annotated screenshot showing OmniParser's grounding output."""
import sys
import time
import pyautogui
from util.utils import get_yolo_model, get_caption_model_processor, get_som_labeled_img, check_ocr_box

out_name = sys.argv[1] if len(sys.argv) > 1 else "annotated_1.png"

print("Loading models...")
yolo = get_yolo_model("weights/icon_detect/model.pt")
caption = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

print("Capturing screen in 3 seconds...")
time.sleep(3)
screenshot_path = "temp_capture.png"
pyautogui.screenshot().save(screenshot_path)

print("Running OmniParser grounding...")
ocr_bbox_rslt, _ = check_ocr_box(screenshot_path, display_img=False, output_bb_format="xyxy", use_paddleocr=False)
text, ocr_bbox = ocr_bbox_rslt

labeled_img, label_coordinates, parsed_content = get_som_labeled_img(
    screenshot_path, yolo, BOX_TRESHOLD=0.05,
    output_coord_in_ratio=False, ocr_bbox=ocr_bbox,
    draw_bbox_config={"text_scale": 0.8, "text_thickness": 2, "text_padding": 3, "thickness": 3},
    caption_model_processor=caption, ocr_text=text, use_local_semantics=True
)

import base64
from PIL import Image
import io
img_data = base64.b64decode(labeled_img)
img = Image.open(io.BytesIO(img_data))
img.save(out_name)
print("Saved:", out_name)