\# Vision-Based Desktop Automation with Dynamic Icon Grounding



TJM Labs take-home assignment — automates locating the Notepad desktop icon

purely by sight, launching it, and typing/saving 10 blog posts fetched from

an API.



\## What this does



For each of 10 posts:

1\. Takes a fresh screenshot of the desktop

2\. Visually grounds (locates) the Notepad icon using OmniParser v2 — no

&#x20;  hardcoded coordinates, no window titles, just looking at the screen

3\. Double-clicks to launch Notepad

4\. Types the post into the editor

5\. Saves it as `post\_{id}.txt` and closes Notepad



The icon is re-grounded fresh on every iteration, not cached after the first

detection — this proves the grounding is genuinely vision-based each time.



\## Grounding approach



Built on Microsoft OmniParser v2 (icon detection + OCR), applying the

coarse-to-fine search-area-reduction strategy from the required paper,

ScreenSpot-Pro (Li et al., 2025, arXiv:2504.07981). See TJM\_Design\_Document.docx

for the full design writeup.



\## Setup



conda create -n omni python=3.12 -y

conda activate omni

pip install -r requirements.txt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

python -c "from huggingface\_hub import snapshot\_download; snapshot\_download(repo\_id='microsoft/OmniParser-v2.0', local\_dir='weights')"



After downloading weights, rename the caption folder if needed:

rename weights\\icon\_caption icon\_caption\_florence



\## Running



python automate.py



The Notepad icon must be visible on the desktop when the script starts.

Output files are written to Desktop/tjm-project/.



\## Known environment notes



\- Tested on Windows with an NVIDIA GPU (CUDA-accelerated PyTorch). The

&#x20; pipeline also runs on CPU-only machines, just slower per grounding pass.

\- This repo's bundled transformers/paddleocr versions had compatibility

&#x20; issues with the OmniParser caption model on a fresh install (Florence-2's

&#x20; forced\_bos\_token\_id attribute error, and a flash\_attn import check with

&#x20; no actual Windows-compatible build). Fixed by pinning

&#x20; transformers==4.38.2 and stubbing flash\_attn as a no-op import, since

&#x20; it's not actually required for inference here. PaddleOCR's constructor was

&#x20; also incompatible with the bundled version and was disabled in favor of

&#x20; EasyOCR, which is what's actually used for the OCR-based grounding signal.

\- On a CPU-only machine, the unattended 10-post loop was timing/focus

&#x20; sensitive (Notepad occasionally didn't have window focus by the time the

&#x20; next action fired). On GPU hardware this did not reproduce across a full

&#x20; clean run.



\## Files



\- automate.py — main automation script

\- posts\_cache.json — local fallback if the JSONPlaceholder API is

&#x20; unreachable

\- TJM\_Design\_Document.docx — design document (objective, architecture,

&#x20; grounding approach, evaluation against the reference paper)

\- tjm-project/ — output folder for the 10 generated post files

