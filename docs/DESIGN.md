# Design Document — Vision-Based Desktop Automation

**TJM Labs Take-Home Assignment**

---

## Table of Contents

1. [Objective](#1-objective)
2. [Assumptions](#2-assumptions)
3. [Architecture](#3-architecture)
4. [Grounding Approach](#4-grounding-approach)
5. [Implementation Details](#5-implementation-details)
6. [Output](#6-output)
7. [Evaluation Against the Reference Paper](#7-evaluation-against-the-reference-paper)
8. [Tradeoffs and Future Work](#8-tradeoffs-and-future-work)
9. [Reference](#9-reference)

---

## 1. Objective

Build a desktop automation program that locates the Notepad application icon on a Windows desktop **purely by sight** — without relying on hardcoded coordinates, window titles, or OS-level shortcuts — then launches Notepad, retrieves ten blog posts from a public API, types each post into the editor, and saves each as a separate text file.

The visual grounding step is the core deliverable: the program must locate the icon the same way a human would, by looking at a screenshot.

---

## 2. Assumptions

- The Notepad icon is visible on the **unoccluded desktop** at the start of each run (not minimized, not buried in a folder, not hidden behind another window).
- The desktop locale is **English** — the OCR text bonus matches the label "Notepad"; non-English locales will lose that signal and rely on CLIP similarity alone.
- A **single display** is connected; multi-monitor setups may produce screenshots that include off-primary regions and confuse the region proposal stage.
- **No Notepad window is already open** when the script starts; an existing open window may satisfy the `_wait_for_notepad` poll immediately, causing the wrong window to receive typed text.
- Screen resolution and icon layout may vary between machines; the grounding step must not depend on fixed pixel coordinates.
- Network access to the post API may be intermittent; a local cache provides a robust fallback so a transient outage does not block the run.
- The program runs unattended once started — no manual clicking is required mid-run.

---

## 3. Architecture

The system is composed of four stages that run once per post, in a loop of ten iterations:

| Stage | Description |
|-------|-------------|
| **Capture** | Take a fresh screenshot of the full desktop |
| **Ground** | Run the screenshot through the visual grounding pipeline to locate the Notepad icon and return its on-screen coordinates |
| **Act** | Double-click the located coordinates to launch Notepad, then type the fetched post text into the editor |
| **Persist** | Save the typed content as `post_{id}.txt` and close Notepad before the next iteration |

Re-grounding on every iteration (rather than caching the icon location after the first detection) was a deliberate choice: it demonstrates that the grounding step is genuinely vision-based and would continue to work even if the icon moved between cycles.

---

## 4. Grounding Approach

The grounding stage is built on **Microsoft's OmniParser v2**, an open-source screen-parsing model that combines an icon detector with an OCR layer to produce a structured list of interactable screen elements and their bounding boxes.

OmniParser was chosen over a paid vision API for three reasons:

1. It runs locally with no per-call cost.
2. It is directly benchmarked in the ScreenSpot-Pro paper referenced below.
3. It provides interpretable intermediate output (bounding boxes and labels) that can be inspected and debugged, rather than a black-box click coordinate.

### Coarse-to-Fine Strategy

The implementation follows the core insight of the required paper — *ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use* (Li et al., 2025) — which observes that grounding accuracy on high-resolution screens improves substantially when the search is narrowed from the full screen to a focused region before fine localization.

The pipeline applies this as a two-pass coarse-to-fine strategy:

1. **Coarse pass** — parse the full screenshot to identify the general desktop icon region.
2. **Fine pass** — re-examine that narrower crop to pinpoint the exact icon centre.

The OCR layer's text recognition (matching the label "Notepad" under the icon) is used as the primary disambiguation signal once the icon-shaped region is found, since icon glyphs alone are visually similar across applications.

---

## 5. Implementation Details

### 5.1 Environment

- **Python 3.12**, managed via a dedicated Conda environment to isolate dependencies from the system Python.
- **PyTorch** with CUDA acceleration where a compatible GPU is available, with automatic fallback to CPU — the grounding logic is identical either way, only inference latency differs.
- **pyautogui** for screen capture, mouse control, and keyboard input.

### 5.2 Data Source

Posts are fetched from the [JSONPlaceholder REST API](https://jsonplaceholder.typicode.com/posts). The fetch is retried up to three times on failure; if the network is unavailable after retries, the program falls back to a local `posts_cache.json` populated from a previous successful fetch, so a transient connectivity issue does not stop the run.

### 5.3 Robustness

- Each of the ten post-processing cycles is wrapped independently, so a failure on one post is logged and the loop continues rather than terminating the entire run.
- The editor is explicitly cleared (select-all, delete) before typing begins, preventing content from one cycle bleeding into the next.
- The run prints a final summary line (e.g. `Done. X/10 posts processed.`) so the outcome of an unattended run is always visible.

---

## 6. Output

Each successful cycle produces one file, `post_{id}.txt`, written to `Desktop/tjm-project/`, containing the title and body of the corresponding blog post exactly as typed into Notepad.

---

## 7. Evaluation Against the Reference Paper

*ScreenSpot-Pro* evaluates grounding models on professional, high-resolution screenshots and finds that most general-purpose vision-language models struggle with small UI elements at full resolution, while accuracy improves significantly when the search space is reduced before fine localization.

This project's two-pass coarse-to-fine design is a direct, small-scale application of that finding: rather than asking the model to localize a small icon within an entire 1080p+ desktop in one shot, the search is narrowed first, then refined.

---

## 8. Tradeoffs and Future Work

### Tradeoffs

| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| **Grounding model** | CLIP zero-shot + region proposals (local) | GPT-4V / paid vision API | Free, runs offline, fully interpretable intermediate output (bounding boxes + scores) |
| **Zero-shot vs fine-tuned** | CLIP zero-shot | Fine-tuned icon detector | Generalises to any desktop icon without training data; lower peak accuracy is an acceptable tradeoff |
| **Region proposals vs YOLO** | Multi-strategy proposals (grid + contour + MSER) | YOLO icon detector | No training data required; zero-shot with respect to icon identity, matching the paper's insight |
| **Local inference vs cloud** | Local OmniParser / CLIP | GPT-4o, Gemini Vision | No per-call cost, no network dependency, no data-privacy concern |
| **GPU vs CPU** | Auto-detect, fall back to CPU | GPU-only | Portability over latency |
| **Grounding frequency** | Re-ground every iteration | Cache coordinates after first detection | Proves genuine vision-based grounding; robust to icon movement between cycles |

### Future Work

- Detect multiple desktop icons and disambiguate among them.
- Explicit handling of small / medium / large Windows icon sizes.
- Support for both light and dark desktop themes.
- Confidence-thresholded fallback strategy when grounding is uncertain.
- Cache the coarse region between cycles when the desktop layout is static (reduces latency at scale).
- Down-scale the first coarse pass while keeping the fine pass at full resolution (throughput optimisation).

---

## 9. Reference

Li, K., Meng, Z., Lin, H., Luo, Z., Tian, Y., Ma, J., Huang, Z., & Chua, T.-S. (2025).
*ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use.*
arXiv:2504.07981. <https://arxiv.org/pdf/2504.07981>
