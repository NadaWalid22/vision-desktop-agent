# Vision-Based Desktop Automation with Dynamic Icon Grounding

TJM Labs take-home assignment — automates locating the Notepad desktop icon purely by sight, launching it, and typing/saving 10 blog posts fetched from a public API.

---

## What It Does

For each of 10 posts fetched from [JSONPlaceholder](https://jsonplaceholder.typicode.com/posts):

1. Takes a fresh screenshot of the desktop (`Win + D` first to ensure the desktop is visible)
2. Visually grounds the Notepad icon using OmniParser v2 — no hardcoded coordinates, no window titles, just looking at the screen
3. Double-clicks to launch Notepad
4. Types the post into the editor
5. Saves it as `post_{id}.txt` and closes Notepad

The icon is re-grounded on every iteration rather than cached after first detection — this proves the grounding is genuinely vision-based each time.

---

## Grounding Approach

Built on **Microsoft OmniParser v2** (YOLO icon detector + Florence-2 captioner + EasyOCR), applying the coarse-to-fine search-area-reduction strategy from the required paper:

> Li et al. (2025). *ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use.* arXiv:2504.07981.

The pipeline runs two stages per grounding attempt:
1. **EasyOCR pass** — fast; catches the "Notepad" text label on the taskbar or desktop
2. **YOLO → Florence-2 pass** — if OCR misses it, detects icon bounding boxes and captions each crop, then matches against "notepad"

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design writeup.

---

## Setup

**Requirements:** Windows, Python 3.12, Conda

```bash
# 1. Create environment
conda create -n omni python=3.12 -y
conda activate omni

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU-only alternative:
# pip install torch torchvision

# 4. Download OmniParser v2 weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='microsoft/OmniParser-v2.0', local_dir='weights')
"

# 5. Rename caption weights folder
mv weights/icon_caption weights/icon_caption_florence
```

---

## Running

```bash
conda activate omni
python automate.py
```

Make sure the Notepad icon is visible on the desktop before starting. Output files are written to `Desktop/tjm-project/`.

---

## Environment Notes

- Tested on Windows with an NVIDIA GPU (CUDA-accelerated PyTorch). The pipeline also runs on CPU-only machines — inference is slower but the grounding logic is identical.
- EasyOCR is used for the OCR grounding signal; PaddleOCR is disabled due to version incompatibility with the bundled dependencies.
- On CPU-only machines, the 10-post loop can be timing/focus sensitive (Notepad occasionally misses window focus before the next action fires). This did not reproduce on GPU hardware across a full clean run.

---

## Repo Structure

```
automate.py            — main automation script
requirements.txt       — Python dependencies
docs/DESIGN.md         — design document (objective, architecture, grounding approach, tradeoffs)
src/grounding/         — region proposal detector (contour, MSER, grid strategies)
util/utils.py          — OmniParser utilities (YOLO inference, Florence-2 captioning, OCR)
weights/               — model checkpoints (not committed; download via setup step 4)
tjm-project/           — output folder for the 10 generated post files
posts_cache.json       — local fallback if the JSONPlaceholder API is unreachable
```
