"""
Grounding package — dynamic visual element localisation.

Public API::

    from src.grounding import VisualGrounder

    grounder = VisualGrounder()
    result   = grounder.locate(screenshot_bgr, "Notepad application icon")
    if result:
        print(result.x, result.y, result.confidence)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.grounding.detector import RegionProposalDetector
from src.grounding.embeddings import CLIPEmbeddingExtractor
from src.grounding.ranking import CandidateRanker, DetectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["VisualGrounder", "DetectionResult"]


class VisualGrounder:
    """
    High-level facade for the full grounding pipeline:

        screenshot -> proposals -> CLIP + OCR scoring -> best match

    Designed to be instantiated once per application session; the CLIP
    model is loaded on first call and kept in memory.

    Args:
        confidence_threshold: Minimum composite score to accept a detection.
        max_proposals: Hard cap on region proposals evaluated per screenshot.
        use_ocr: Whether to run OCR for text-bonus scoring.
        clip_query_template: f-string with {name} placeholder for query expansion.
    """

    _DEFAULT_QUERY_TEMPLATE = (
        "a {name} application icon on a Windows desktop, small square icon"
    )

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        max_proposals: int = 200,
        use_ocr: bool = True,
        clip_query_template: str | None = None,
    ) -> None:
        self._extractor = CLIPEmbeddingExtractor()
        self._detector = RegionProposalDetector(max_proposals=max_proposals)
        self._ranker = CandidateRanker(
            extractor=self._extractor,
            confidence_threshold=confidence_threshold,
            use_ocr=use_ocr,
        )
        self._query_template = clip_query_template or self._DEFAULT_QUERY_TEMPLATE

    def locate(
        self,
        screenshot: "NDArray",
        target_name: str,
        query: str | None = None,
        top_k: int = 5,
    ) -> DetectionResult | None:
        """
        Locate *target_name* in *screenshot*.

        Args:
            screenshot: BGR uint8 numpy array (H, W, 3).
            target_name: Short, human-readable label, e.g. "Notepad".
            query: Override the full CLIP query. If None, uses the template.
            top_k: Number of candidates to return internally (only best is returned).

        Returns:
            DetectionResult with pixel coordinates and confidence, or None
            if no match exceeded the threshold.
        """
        t0 = time.perf_counter()

        effective_query = query or self._query_template.format(name=target_name)
        logger.debug(f"Grounding query: '{effective_query}'")

        proposals = self._detector.propose(screenshot)
        logger.debug(f"Generated {len(proposals)} region proposals")

        ranked = self._ranker.rank(proposals, effective_query, target_name)
        elapsed = time.perf_counter() - t0

        if ranked:
            best = ranked[0]
            logger.info(
                f"Located '{target_name}' at ({best.x}, {best.y}) "
                f"[conf={best.confidence:.3f}] in {elapsed:.2f}s"
            )
            return best

        logger.warning(
            f"Could not locate '{target_name}' "
            f"(threshold={self._ranker.confidence_threshold}) "
            f"in {elapsed:.2f}s"
        )
        return None

    def locate_all(
        self,
        screenshot: "NDArray",
        target_name: str,
        query: str | None = None,
    ) -> list[DetectionResult]:
        """
        Return ALL candidates for *target_name*, sorted by confidence descending.

        Unlike locate(), which silently returns only the top hit, this exposes
        the full ranked list so callers can handle the multiple-match case
        (e.g. two Notepad icons on screen) explicitly.
        """
        effective_query = query or self._query_template.format(name=target_name)
        proposals = self._detector.propose(screenshot)
        return self._ranker.rank(proposals, effective_query, target_name)

    def locate_with_retry(
        self,
        screenshot_fn,
        target_name: str,
        retries: int = 3,
        confidence_decay: float = 0.05,
        query: str | None = None,
    ) -> DetectionResult | None:
        """
        Attempt detection with progressively relaxed confidence thresholds.

        On each failed attempt we capture a fresh screenshot (icons may
        animate) and lower the threshold by *confidence_decay*. This
        implements the retry strategy described in the design document.
        """
        original_threshold = self._ranker.confidence_threshold

        for attempt in range(1, retries + 1):
            screenshot = screenshot_fn()
            result = self.locate(screenshot, target_name, query)
            if result:
                self._ranker.confidence_threshold = original_threshold
                return result

            # Relax threshold for next attempt
            self._ranker.confidence_threshold = max(
                0.10, original_threshold - confidence_decay * attempt
            )
            logger.warning(
                f"Retry {attempt}/{retries} — "
                f"lowering threshold to {self._ranker.confidence_threshold:.2f}"
            )

        self._ranker.confidence_threshold = original_threshold
        return None
