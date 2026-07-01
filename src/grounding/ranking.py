"""
Semantic ranking module — the "grounding" half of the pipeline.

Given a list of RegionProposal objects and a natural-language query,
this module scores every candidate using a hybrid signal:

    score = α · clip_similarity + β · ocr_bonus + γ · aspect_bonus

1. **CLIP similarity** (α = 0.7)
   Cosine similarity between the region's CLIP image embedding and the
   query text embedding.  This is the primary signal and is what makes
   the system zero-shot: it generalises to icons it has never seen before.

2. **OCR text bonus** (β = 0.3)
   If pytesseract detects text in the region that matches the target name
   (case-insensitive), we add a positive bonus.  Icon labels (e.g. "Notepad"
   written below the icon) are highly discriminative.

3. **Aspect-ratio bonus** (γ = 0.05)
   Desktop icons are roughly square (0.8–1.2 AR).  Slightly penalise
   extremely tall or wide proposals to reduce false positives on taskbar
   buttons, window title bars, etc.

After scoring, we apply a minimum confidence threshold and return
a ranked list of DetectionResult objects.

This architecture directly mirrors the "rank-then-ground" formulation in
GUI Agents with Dynamic Grounding: the model doesn't need to know the exact
pixel position at training time — it reasons semantically at inference time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from src.grounding.detector import BoundingBox, RegionProposal
from src.grounding.embeddings import CLIPEmbeddingExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Scoring weights ──────────────────────────────────────────────────────────
_W_CLIP = 0.70
_W_OCR = 0.25
_W_ASPECT = 0.05


@dataclass
class DetectionResult:
    """
    The output of the grounding pipeline: a grounded element with its
    screen coordinates and confidence score.
    """

    x: int           # pixel x of the bounding-box centre
    y: int           # pixel y of the bounding-box centre
    confidence: float  # 0.0–1.0

    box: BoundingBox
    clip_score: float
    ocr_score: float
    detected_text: str

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "confidence": round(self.confidence, 4),
            "box": self.box.as_tuple(),
            "clip_score": round(self.clip_score, 4),
            "ocr_score": round(self.ocr_score, 4),
            "detected_text": self.detected_text,
        }


class CandidateRanker:
    """
    Scores and ranks region proposals against a text query.

    Args:
        extractor: Shared CLIPEmbeddingExtractor instance.
        confidence_threshold: Minimum score to include a result.
        use_ocr: Whether to run pytesseract on each proposal.
    """

    def __init__(
        self,
        extractor: CLIPEmbeddingExtractor | None = None,
        confidence_threshold: float = 0.25,
        use_ocr: bool = True,
    ) -> None:
        self._extractor = extractor or CLIPEmbeddingExtractor()
        self.confidence_threshold = confidence_threshold
        self.use_ocr = use_ocr
        self._ocr_available = self._check_ocr()

    def rank(
        self,
        proposals: list[RegionProposal],
        query: str,
        target_name: str | None = None,
    ) -> list[DetectionResult]:
        """
        Score all proposals against *query* and return sorted DetectionResults.

        Args:
            proposals:   Candidate regions from the detector.
            query:       Full natural-language description for CLIP matching.
                         E.g. "Notepad text editor icon on a Windows desktop".
            target_name: Short name used for OCR keyword matching, e.g. "notepad".
                         If None, derived from the first word of query.

        Returns:
            List of DetectionResult, sorted by confidence (highest first),
            filtered to >= confidence_threshold.
        """
        if not proposals:
            logger.warning("No proposals to rank — detector returned empty list")
            return []

        if target_name is None:
            target_name = query.split()[0].lower()

        # Pre-encode the query text once (cached)
        text_emb = self._extractor.encode_text_cached(query)

        results: list[DetectionResult] = []

        for prop in proposals:
            try:
                result = self._score_proposal(prop, text_emb, target_name)
                if result.confidence >= self.confidence_threshold:
                    results.append(result)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Skipping proposal {prop.box}: {e}")

        results.sort(key=lambda r: r.confidence, reverse=True)

        if results:
            best = results[0]
            logger.info(
                f"Top match: ({best.x}, {best.y}) "
                f"conf={best.confidence:.3f} "
                f"clip={best.clip_score:.3f} "
                f"ocr='{best.detected_text}'"
            )
        else:
            logger.warning(
                f"No proposals exceeded confidence threshold {self.confidence_threshold}"
            )

        return results

    def best(
        self,
        proposals: list[RegionProposal],
        query: str,
        target_name: str | None = None,
    ) -> DetectionResult | None:
        """Convenience wrapper — returns only the top result or None."""
        ranked = self.rank(proposals, query, target_name)
        return ranked[0] if ranked else None

    # ─── Internal scoring ──────────────────────────────────────────────────────

    def _score_proposal(
        self,
        prop: RegionProposal,
        text_emb: np.ndarray,
        target_name: str,
    ) -> DetectionResult:
        # 1. CLIP similarity
        img_emb = self._extractor.encode_image(prop.crop)
        clip_score = self._extractor.cosine_similarity(img_emb, text_emb)
        # CLIP raw scores are typically in [0, 1] after normalisation;
        # remap from [-1, 1] range: (score + 1) / 2
        clip_score_norm = (clip_score + 1.0) / 2.0

        # 2. OCR bonus
        ocr_score, detected_text = self._ocr_score(prop.crop, target_name)

        # 3. Aspect-ratio bonus (icon ≈ square)
        ar = prop.box.w / max(prop.box.h, 1)
        aspect_bonus = max(0.0, 1.0 - abs(ar - 1.0))  # 1 when square, 0 when 2:1

        # Weighted combination
        composite = (
            _W_CLIP * clip_score_norm
            + _W_OCR * ocr_score
            + _W_ASPECT * aspect_bonus
        )

        return DetectionResult(
            x=prop.box.cx,
            y=prop.box.cy,
            confidence=min(composite, 1.0),
            box=prop.box,
            clip_score=clip_score_norm,
            ocr_score=ocr_score,
            detected_text=detected_text,
        )

    def _ocr_score(self, crop: np.ndarray, target_name: str) -> tuple[float, str]:
        """
        Run OCR on the crop and return (score 0–1, detected text).
        Score = 1.0 if target_name substring found (case-insensitive), else 0.
        Gracefully degrades if pytesseract is unavailable.
        """
        if not self._ocr_available or not self.use_ocr:
            return 0.0, ""

        try:
            import pytesseract  # type: ignore[import]
            from PIL import Image

            pil = Image.fromarray(crop[..., ::-1])  # BGR → RGB
            # Enlarge tiny crops for better OCR accuracy
            if pil.width < 64 or pil.height < 64:
                scale = max(64 // pil.width, 64 // pil.height, 1)
                pil = pil.resize(
                    (pil.width * scale, pil.height * scale),
                    Image.LANCZOS,
                )

            config = "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
            text = pytesseract.image_to_string(pil, config=config).strip().lower()

            # Fuzzy substring match (handles "Notepad" → "notepad", "note pad", etc.)
            pattern = re.sub(r"\s+", r"\\s*", re.escape(target_name.lower()))
            match = bool(re.search(pattern, text))
            return (1.0 if match else 0.0), text

        except Exception as e:  # noqa: BLE001
            logger.debug(f"OCR error: {e}")
            return 0.0, ""

    @staticmethod
    def _check_ocr() -> bool:
        try:
            import pytesseract  # noqa: F401 # type: ignore[import]
            return True
        except ImportError:
            logger.warning(
                "pytesseract not found — OCR bonus disabled. "
                "Install it with: pip install pytesseract"
            )
            return False
