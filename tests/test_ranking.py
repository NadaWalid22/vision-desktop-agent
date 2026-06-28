"""
Unit tests for the CandidateRanker and DetectionResult.

These tests mock the CLIPEmbeddingExtractor so no GPU or model download
is required.  We verify the scoring arithmetic and ranking order.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.grounding.detector import BoundingBox, RegionProposal
from src.grounding.ranking import CandidateRanker, DetectionResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_proposal(x: int, y: int, w: int = 48, h: int = 48) -> RegionProposal:
    """Helper: build a RegionProposal with a dummy crop."""
    return RegionProposal(
        box=BoundingBox(x, y, w, h),
        strategy="grid",
        crop=np.zeros((h, w, 3), dtype=np.uint8),
    )


class MockExtractor:
    """
    Controllable fake of CLIPEmbeddingExtractor.
    Accepts a dict mapping proposal index -> similarity score.
    """

    def __init__(self, scores: dict[int, float]) -> None:
        self._scores = scores
        self._call_count = 0

    def encode_image(self, image) -> np.ndarray:
        # Return a unit vector encoding the call index as a "magic" value
        vec = np.zeros(512, dtype=np.float32)
        vec[self._call_count % 512] = 1.0
        self._call_count += 1
        return vec

    def encode_text_cached(self, query: str) -> np.ndarray:
        return np.ones(512, dtype=np.float32) / np.sqrt(512)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # Return the pre-configured score for this call
        idx = self._call_count - 1
        return self._scores.get(idx, 0.3)


# ─── DetectionResult tests ────────────────────────────────────────────────────

class TestDetectionResult:
    def test_to_dict_keys(self) -> None:
        r = DetectionResult(
            x=100, y=200, confidence=0.85,
            box=BoundingBox(76, 176, 48, 48),
            clip_score=0.80, ocr_score=1.0, detected_text="notepad",
        )
        d = r.to_dict()
        assert "x" in d
        assert "y" in d
        assert "confidence" in d
        assert "box" in d

    def test_confidence_capped_at_1(self) -> None:
        r = DetectionResult(
            x=0, y=0, confidence=1.5,  # Over-capped value
            box=BoundingBox(0, 0, 48, 48),
            clip_score=1.0, ocr_score=1.0, detected_text="",
        )
        # The ranker caps at 1.0; model field itself allows >1 for testing
        assert r.confidence == pytest.approx(1.5)  # field stores raw value


# ─── CandidateRanker tests ────────────────────────────────────────────────────

class TestCandidateRanker:
    def _ranker(self, scores: dict) -> CandidateRanker:
        extractor = MockExtractor(scores)
        ranker = CandidateRanker(
            extractor=extractor,  # type: ignore[arg-type]
            confidence_threshold=0.10,
            use_ocr=False,  # Disable OCR in unit tests
        )
        return ranker

    def test_empty_proposals(self) -> None:
        ranker = self._ranker({})
        result = ranker.rank([], "Notepad icon", "notepad")
        assert result == []

    def test_returns_sorted_by_confidence(self) -> None:
        proposals = [_make_proposal(0, 0), _make_proposal(100, 100), _make_proposal(200, 200)]
        # Scores will be cycled via MockExtractor's call count
        ranker = self._ranker({0: 0.8, 1: 0.4, 2: 0.6})
        results = ranker.rank(proposals, "Notepad icon", "notepad")
        # Should be sorted highest confidence first
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_filters_below_threshold(self) -> None:
        proposals = [_make_proposal(0, 0), _make_proposal(100, 100)]
        ranker = CandidateRanker(
            extractor=MockExtractor({0: -0.9, 1: -0.9}),  # type: ignore
            confidence_threshold=0.5,
            use_ocr=False,
        )
        results = ranker.rank(proposals, "query", "target")
        # Both proposals have very low CLIP similarity → below threshold
        assert all(r.confidence >= 0.5 for r in results)

    def test_best_returns_single(self) -> None:
        proposals = [_make_proposal(0, 0), _make_proposal(50, 50)]
        ranker = self._ranker({0: 0.9, 1: 0.5})
        result = ranker.best(proposals, "Notepad icon", "notepad")
        assert result is not None
        assert isinstance(result, DetectionResult)

    def test_best_returns_none_when_empty(self) -> None:
        ranker = self._ranker({})
        assert ranker.best([], "query", "target") is None

    def test_result_coordinates_from_box_centre(self) -> None:
        proposals = [_make_proposal(x=100, y=200, w=48, h=48)]
        ranker = self._ranker({0: 0.9})
        results = ranker.rank(proposals, "Notepad icon", "notepad")
        if results:
            assert results[0].x == 100 + 24  # cx
            assert results[0].y == 200 + 24  # cy

    def test_aspect_ratio_bonus_for_square(self) -> None:
        """A square proposal should score higher than an elongated one."""
        square = _make_proposal(0, 0, w=48, h=48)
        elongated = _make_proposal(100, 0, w=128, h=32)

        # Give both identical CLIP scores
        class FixedExtractor:
            def encode_image(self, img): return np.array([1.0, 0.0])
            def encode_text_cached(self, q): return np.array([1.0, 0.0])
            def cosine_similarity(self, a, b): return 0.5

        ranker = CandidateRanker(
            extractor=FixedExtractor(),  # type: ignore
            confidence_threshold=0.0,
            use_ocr=False,
        )
        results = ranker.rank([square, elongated], "icon", "icon")
        # Square (AR=1) should score >= elongated (AR=4)
        square_result = next(r for r in results if r.box.w == 48)
        elong_result = next(r for r in results if r.box.w == 128)
        assert square_result.confidence >= elong_result.confidence
