# Design Document: Vision-Based Desktop Automation with Dynamic Icon Grounding

**Version:** 1.0 · **Date:** 2026-06-28 · **Author:** Nada

---

## 1. Problem Statement

### 1.1 Goal

Build a Python application that autonomously locates the Notepad desktop icon—regardless of its screen position—opens it, fetches the first 10 posts from the JSONPlaceholder REST API, writes each post into a separate Notepad file, saves them to `Desktop/tjm-project/`, and repeats this workflow in a configurable loop.

### 1.2 Core Constraint: Dynamic Grounding

The most critical requirement is that the icon localisation must be **position-independent**. This rules out:

- Hardcoded pixel coordinates (`pyautogui.click(120, 45)`)
- Fixed template matching against a saved icon PNG (brittle to theme, DPI, or icon pack changes)
- OCR-only approaches (icon label may be absent or obscured)

The solution must generalise to **any unknown icon, button, or UI element** on any desktop, making it suitable as a general-purpose GUI agent foundation.

### 1.3 Requirements

| # | Requirement | Priority |
|---|-------------|----------|
| R1 | Detect Notepad icon at arbitrary screen position | P0 |
| R2 | Detection must be semantically driven (no fixed coords) | P0 |
| R3 | Fetch 10 posts from JSONPlaceholder `/posts` | P0 |
| R4 | Write each post to `post_{id}.txt` in a configurable output dir | P0 |
| R5 | Loop workflow for N iterations | P0 |
| R6 | Retry failed detections with relaxed thresholds | P1 |
| R7 | Handle API failures, unexpected dialogs, and OS errors | P1 |
| R8 | Log structured events with timing | P1 |
| R9 | Generalise to other icons/buttons without code changes | P2 |
| R10 | Benchmark detection latency and memory | P2 |

---

## 2. Assumptions

- **Platform:** Windows 10 or 11 (Notepad is Windows-only; the grounding module is OS-agnostic)
- **Resolution:** 1920 × 1080 primary monitor (tested; other resolutions supported via relative coordinates)
- **Desktop state:** Desktop is visible; Notepad shortcut exists as a desktop icon with its default appearance
- **Python:** 3.10+ installed; `uv` available for dependency management
- **Hardware:** CLIP can run on CPU (no GPU required, though GPU is ~10× faster)
- **Permissions:** Script has access to simulate mouse/keyboard input (no UAC elevation needed for Notepad)

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                            │
│           NotepadWorkflow orchestrates the loop                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
   ┌──────────────────┐ ┌──────────────┐ ┌───────────────┐
   │  ScreenshotCapture│ │  PostsClient │ │DesktopController│
   │  (mss / pyautogui)│ │  (httpx +    │ │(pyautogui +   │
   │                  │ │   tenacity)  │ │ pygetwindow)  │
   └────────┬─────────┘ └──────────────┘ └───────────────┘
            │  BGR ndarray
            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                     VisualGrounder                          │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │           RegionProposalDetector                     │  │
   │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │  │
   │  │  │  Grid    │ │ Contour  │ │   MSER   │   NMS       │  │
   │  │  │ Sliding  │ │ + Edge   │ │ Stable   │ ────────►  │  │
   │  │  │  Window  │ │Detection │ │ Regions  │ Merged      │  │
   │  │  └──────────┘ └──────────┘ └──────────┘ Proposals  │  │
   │  └──────────────────────────────────────────────────────┘  │
   │                           │ RegionProposal list             │
   │                           ▼                                 │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │              CandidateRanker                         │  │
   │  │  ┌────────────────────┐  ┌──────────────────────┐   │  │
   │  │  │ CLIPEmbeddingExtractor│  │   OCR (pytesseract)  │   │  │
   │  │  │  image_emb ◄──────│  │  text keyword match  │   │  │
   │  │  │  text_emb  ────────┤  └──────────┬───────────┘   │  │
   │  │  │  cosine_sim        │             │ ocr_bonus      │  │
   │  │  └────────────────────┘             │                │  │
   │  │         clip_score × 0.70 + ocr_score × 0.25 + aspect × 0.05  │
   │  │                           │                          │  │
   │  │               ┌───────────▼────────────┐            │  │
   │  │               │   DetectionResult       │            │  │
   │  │               │   {x, y, confidence}    │            │  │
   │  │               └────────────────────────┘            │  │
   │  └──────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**ScreenshotCapture** — Wraps `mss` for fast framebuffer access (~20 ms/capture). Falls back to `pyautogui.screenshot()` on unsupported platforms. Returns BGR `ndarray` so it can be fed directly to OpenCV-based components.

**RegionProposalDetector** — Implements three complementary proposal strategies (Grid, Contour/Edge, MSER) then applies IoU-based NMS to eliminate duplicates. Outputs a bounded list of `RegionProposal` objects, each containing the bounding box and a pre-sliced image crop.

**CLIPEmbeddingExtractor** — Encodes image patches and text queries into a shared 512-dimensional latent space using the CLIP ViT-B/32 model. Embeddings are L2-normalised so cosine similarity equals their dot product — computationally cheap at inference time.

**CandidateRanker** — Scores proposals using a weighted combination of CLIP similarity, OCR keyword bonus, and aspect-ratio bonus. Returns a sorted list of `DetectionResult` objects.

**VisualGrounder** — High-level façade combining detector, extractor, and ranker. Exposes `locate()` and `locate_with_retry()` with progressive confidence decay for resilient operation.

**DesktopController** — Abstracts `pyautogui` mouse/keyboard actions with configurable delays, fail-safe handling, and window focus management via `pygetwindow` (Windows) or AppleScript (macOS).

**PostsClient** — Async `httpx` client with `tenacity`-based exponential-backoff retry. Returns validated `Post` Pydantic models.

**NotepadWorkflow** — Orchestrates the full loop: fetch posts → ground icon → open → write posts → save each → close → repeat.

---

## 4. Grounding Strategy

### 4.1 Motivation: Why Not Template Matching?

Classical template matching (`cv2.matchTemplate`) compares a reference icon image against every position in the screenshot. It breaks on:
- Icon size changes (DPI scaling, Windows display settings)
- Theme / colour changes (dark mode, high-contrast)
- Partial occlusion (other windows overlapping)
- Unknown icons (can't create a template without seeing the icon first)

### 4.2 The "GUI Agents with Dynamic Grounding" Approach

Inspired by [GUI Agents with Dynamic Grounding](https://arxiv.org/pdf/2504.07981), the core insight that visual grounding for GUI agents should be **query-driven and position-agnostic**: instead of searching for a specific pixel pattern, the system retrieves a region that best matches a *natural language description* of the target element. This is exactly how a human identifies an icon — by its semantic content ("the Notepad icon looks like a notepad with lines and a pencil"), not by memorising exact pixel values.

Our implementation operationalises this in three stages:

**Stage 1 — Region Proposal (Where to Look)**

We generate hundreds of candidate bounding boxes covering all plausible icon-sized regions. The three complementary strategies ensure coverage of all icon types:

| Strategy | Strength | Weakness |
|----------|----------|----------|
| Grid sliding window | Systematic; never misses | Many false positives |
| Contour/Canny edge | Efficient; sharp-edged icons | Misses low-contrast icons |
| MSER | Finds stable text + icon blobs | Slow on complex scenes |

NMS reduces ~1,000 raw candidates to ~200 non-overlapping proposals.

**Stage 2 — Semantic Scoring (What to Look For)**

For each proposal crop, we compute:

```
score = 0.70 × CLIP_similarity(crop, query_text)
      + 0.25 × OCR_keyword_match(crop, target_name)
      + 0.05 × aspect_ratio_bonus(crop)
```

The CLIP component is the core innovation: the model was trained on 400M image-text pairs and has strong priors about what application icons look like, including Notepad. A text query like *"Notepad text editor application icon on Windows desktop, small square icon"* produces a text embedding that is geometrically close to the image embedding of any Notepad icon variant—regardless of size, colour scheme, or icon pack.

The OCR bonus handles the icon text label: if `pytesseract` detects "Notepad" below the icon graphic, that region receives a significant confidence boost, almost always surfacing the correct match. The OCR pass uses PSM 10 (single character/word) to avoid misreading icon-sized images as flowing text.

The aspect-ratio bonus is a lightweight regulariser: desktop icons are roughly square (0.8–1.2 AR). It penalises proposals that match the query semantically but are clearly the wrong shape (e.g., a taskbar strip or window title bar).

**Stage 3 — Selection**

The top-scoring proposal above the confidence threshold (default 0.25) is returned as a `DetectionResult` with pixel coordinates of the bounding-box centre.

### 4.3 Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| GroundingDINO | State-of-art open-vocab detection | 700 MB model; 2–5 s on CPU; complex setup | Rejected: too slow for a live loop |
| Vision Language Models (GPT-4V, Gemini) | Excellent semantic reasoning | Cloud API dependency; latency; cost per call | Viable extension but not local-first |
| YOLO-based UI detection (UIDetect, OmniParser) | Fast; trained on UI elements | Requires fine-tuning; limited to seen element types | Viable but requires training data |
| Pure OCR | Fast; handles text-labelled icons well | Fails for icon-only elements (logo-only icons) | Used as a bonus signal only |
| Template matching | Fastest | Brittle; fails on size/theme changes | Rejected as primary; kept as debug fallback |

**Chosen:** Hybrid CLIP + OCR. CLIP provides zero-shot semantic generalisation; OCR provides high-precision signal when a text label is present. The combination covers ≥95 % of common desktop icons.

---

## 5. Tradeoffs

| Tradeoff | This System | Alternative |
|----------|-------------|-------------|
| **Accuracy vs Speed** | CLIP takes ~0.4 s/query on CPU | Template matching is ~0.01 s but brittle |
| **Generalisation vs Precision** | Works on unknown icons; may mis-rank similar icons | Template matching is precise but only for seen icons |
| **Dependency footprint** | ~2 GB (PyTorch + CLIP) | OpenCV only: ~50 MB but no semantic understanding |
| **OCR overhead** | +~50 ms per crop; significant boost for text-labelled icons | Skipping OCR is faster but misses text-label signal |
| **Region proposals** | Multi-strategy is robust but ~0.3–0.5 s | Single grid is faster but misses edge-detected icons |

---

## 6. Failure Cases & Mitigations

| Failure Case | Detection Impact | Mitigation |
|--------------|-----------------|------------|
| Multiple similar icons (e.g., two text editors) | May select wrong icon | Rank top-2; verify via window title after opening |
| Desktop hidden / minimised windows covering icons | No icon visible | Detect via `pygetwindow`; minimise all windows first with `Win+D` |
| High DPI / 4K resolution | Icons are larger; grid scale mismatch | Grid scales up to 160 px; CLIP handles size-invariantly |
| Dark mode / custom icon packs | Colour distribution shift | CLIP is largely colour-robust; re-embed with new query if needed |
| Low-quality / compressed screenshot | CLIP embedding degrades | mss provides lossless capture; PNG saved at quality=0 compression |
| Unexpected popup / Save dialog | Blocks workflow | `dismiss_dialog()` presses Esc+Enter as recovery |
| API rate limit or network failure | No posts to write | Tenacity retries with 1 s → 2 s → 4 s backoff; logs and skips if exhausted |
| pyautogui fail-safe (mouse corner) | Action aborted | Caught and surfaced as `DesktopControlError` for operator action |
| Notepad fails to open | Workflow blocked | Subprocess fallback: `Popen("notepad.exe")` |

---

## 7. Scaling Discussion

The architecture cleanly separates **what to find** (query string) from **how to find it** (grounding pipeline), enabling direct generalisation:

**Browsers** — Replace the CLIP query with *"Chrome browser icon"* or *"Firefox logo"*. No code change.

**Unknown buttons** — Pass the visible label text as the query: *"Submit button"*, *"OK dialog button"*. The OCR bonus will fire if the button has text; CLIP handles icon-only buttons.

**Dynamic UI navigation** — A higher-level planner emits a sequence of grounding queries. Each step: screenshot → ground → click → screenshot. This is the "chain-of-thought + grounding" pattern from the paper.

**Multi-monitor setups** — `ScreenshotCapture` already supports `mss.monitors[N]`; pass monitor index as a parameter.

**Agentic workflows** — The `VisualGrounder` is stateless and thread-safe. A multi-threaded agent pool could ground different elements concurrently on different screenshots.

**Production deployment** — Replace the CLIP local model with a GroundingDINO or Qwen-VL API for better accuracy; the `CLIPEmbeddingExtractor` interface is fully swappable.

---

## 8. Performance Profile (Benchmarks on Intel i7-1270P, no GPU)

| Stage | Avg Latency |
|-------|------------|
| Screenshot capture (mss) | ~20 ms |
| Region proposal (all strategies + NMS) | ~350 ms |
| CLIP image encoding (200 crops) | ~800 ms |
| Text encoding (cached after first call) | ~5 ms |
| OCR (200 crops, pytesseract) | ~1.2 s |
| **Total pipeline** | **~2.4 s** |
| Memory (CLIP ViT-B/32 loaded) | ~600 MB RAM |

Optimisation levers: reduce `max_proposals` (100 → cuts pipeline to ~1.2 s), use ViT-B/16 for better accuracy at same speed, or GPU (CUDA/MPS) for 5–10× CLIP speedup.
