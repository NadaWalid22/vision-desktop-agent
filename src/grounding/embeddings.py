"""
CLIP-based visual and text embedding extractor.

This module is the semantic backbone of the grounding pipeline.
We use OpenAI CLIP (via open-clip-torch) to produce aligned image/text
embeddings in a shared latent space, enabling zero-shot icon detection:
the model was never trained to find "Notepad icons" specifically, yet
cosine similarity between an image patch and the text query
"Notepad application icon on Windows desktop" reliably surfaces matches.

Design note: We lazy-load the model on first use and cache it as a
module-level singleton so repeated calls within one workflow don't pay
the GPU-transfer penalty on every screenshot.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Model singleton (thread-safe lazy init) ──────────────────────────────────
_model_lock = threading.Lock()
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai") -> tuple:
    """Load CLIP model, returning (model, preprocess, tokenizer)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer

    with _model_lock:
        if _clip_model is not None:
            return _clip_model, _clip_preprocess, _clip_tokenizer

        logger.info(f"Loading CLIP model '{model_name}' (pretrained='{pretrained}')…")
        try:
            import open_clip  # type: ignore[import]

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            device = _get_device()
            model = model.eval().to(device)

            _clip_model = model
            _clip_preprocess = preprocess
            _clip_tokenizer = tokenizer

            logger.success(f"CLIP loaded on {device}")
        except ImportError:
            logger.warning(
                "open-clip-torch not found — falling back to HuggingFace CLIP. "
                "Install open-clip-torch for better performance."
            )
            from transformers import CLIPModel, CLIPProcessor  # type: ignore[import]

            hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            hf_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            device = _get_device()
            hf_model = hf_model.eval().to(device)

            # Wrap HuggingFace to match the open_clip interface expected below
            _clip_model = _HFCLIPWrapper(hf_model, hf_proc, device)
            _clip_preprocess = _HFImagePreprocess(hf_proc)
            _clip_tokenizer = _HFTokenizerWrapper(hf_proc)

        return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# ─── HuggingFace shim ─────────────────────────────────────────────────────────

class _HFCLIPWrapper:
    """Thin wrapper so HuggingFace CLIP behaves like open_clip for our pipeline."""

    def __init__(self, model, processor, device: torch.device) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self._model.get_image_features(pixel_values=pixel_values.to(self._device))
        return out

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self._model.get_text_features(
            input_ids=input_ids["input_ids"].to(self._device),
            attention_mask=input_ids["attention_mask"].to(self._device),
        )
        return out


class _HFImagePreprocess:
    def __init__(self, processor) -> None:
        self._p = processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self._p(images=image, return_tensors="pt")["pixel_values"].squeeze(0)


class _HFTokenizerWrapper:
    def __init__(self, processor) -> None:
        self._p = processor

    def __call__(self, texts: list[str]) -> dict:
        return self._p(text=texts, return_tensors="pt", padding=True, truncation=True)


# ─── Public API ───────────────────────────────────────────────────────────────

class CLIPEmbeddingExtractor:
    """
    Extracts image and text embeddings using CLIP.

    Usage::

        extractor = CLIPEmbeddingExtractor()
        img_emb   = extractor.encode_image(pil_image)      # (D,)
        txt_emb   = extractor.encode_text("Notepad icon")  # (D,)
        score     = extractor.cosine_similarity(img_emb, txt_emb)  # float

    Thread safety: safe to call from multiple threads (model init is locked).
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = _get_device()

    @property
    def _components(self) -> tuple:
        return _load_clip(self._model_name, self._pretrained)

    def encode_image(self, image: Image.Image | NDArray) -> np.ndarray:
        """
        Encode a PIL image or numpy BGR/RGB array to an L2-normalised embedding.

        Args:
            image: PIL Image or numpy array (HWC, BGR or RGB).

        Returns:
            1-D numpy array of shape (D,), dtype float32, L2-normalised.
        """
        pil = _to_pil(image)
        model, preprocess, _ = self._components

        pixel = preprocess(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            if hasattr(model, "encode_image"):
                feats = model.encode_image(pixel)  # open_clip path
            else:
                feats = model.encode_image(pixel)  # wrapper path

        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).cpu().float().numpy()

    def encode_text(self, query: str) -> np.ndarray:
        """
        Encode a text query to an L2-normalised embedding.

        Args:
            query: Natural-language description of the target element.

        Returns:
            1-D numpy array of shape (D,), dtype float32, L2-normalised.
        """
        model, _, tokenizer = self._components

        tokens = tokenizer([query])
        # Both open_clip and our HF wrapper support .to() on the tokens
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self._device)
        with torch.no_grad():
            if hasattr(model, "encode_text"):
                feats = model.encode_text(tokens)
            else:
                feats = model.encode_text(tokens)

        feats = F.normalize(feats, dim=-1)
        return feats.squeeze(0).cpu().float().numpy()

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalised embeddings."""
        # Both are already normalised so dot product == cosine similarity
        return float(np.dot(a, b))

    @lru_cache(maxsize=32)
    def encode_text_cached(self, query: str) -> np.ndarray:
        """Cache text embeddings — the query rarely changes within one workflow."""
        return self.encode_text(query)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_pil(image: Image.Image | NDArray) -> Image.Image:
    """Convert numpy array (BGR or RGB) to PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    import cv2  # type: ignore[import]
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    return Image.fromarray(rgb.astype(np.uint8))
