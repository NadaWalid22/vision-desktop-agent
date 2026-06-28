"""
Integration tests for the VisualGrounder facade.

Tests the full pipeline (detector → ranker → result) with mocked CLIP
so the test suite runs fast and without GPU.
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest

from src.grounding import VisualGrounder
from src.grounding.detector import BoundingBox, RegionProposal
from src.grounding.ranking import DetectionResult


@pytest.fixture
def synthetic_desktop() -> np.ndarray:
    """1920x1080 dark grey desktop with a bright icon patch at (200, 300)."""
    img = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    img[300:364, 200:264] = [30, 150, 220]  # Icon-coloured square
    return img


@pytest.fixture
def mock_detection_result() -> DetectionResult:
    return DetectionResult(
        x=232, y=332,
        confidence=0.82,
        box=BoundingBox(200, 300, 64, 64),
        clip_score=0.78,
        ocr_score=1.0,
        detected_text="notepad",
    )


class TestVisualGrounder:
    def test_locate_returns_none_when_below_threshold(
        self, synthetic_desktop: np.ndarray
    ) -> None:
        grounder = VisualGrounder(confidence_threshold=0.99)
        with mock.patch.object(
            grounder._ranker, "rank", return_value=[]
        ):
            result = grounder.locate(synthetic_desktop, "Notepad")
            assert result is None

    def test_locate_returns_detection_result(
        self,
        synthetic_desktop: np.ndarray,
        mock_detection_result: DetectionResult,
    ) -> None:
        grounder = VisualGrounder(confidence_threshold=0.1)
        with mock.patch.object(
            grounder._ranker, "rank", return_value=[mock_detection_result]
        ):
            result = grounder.locate(synthetic_desktop, "Notepad")
            assert result is not None
            assert result.x == 232
            assert result.y == 332
            assert result.confidence == pytest.approx(0.82)

    def test_locate_uses_query_template(self, synthetic_desktop: np.ndarray) -> None:
        template = "an icon called {name} on the screen"
        grounder = VisualGrounder(clip_query_template=template)
        captured_query = []

        original_locate = grounder.locate

        def patched_locate(screenshot, target_name, query=None, **kwargs):
            captured_query.append(query or template.format(name=target_name))
            return None

        with mock.patch.object(grounder, "locate", side_effect=patched_locate):
            grounder.locate(synthetic_desktop, "Notepad")

        if captured_query:
            assert "Notepad" in captured_query[0]

    def test_locate_with_retry_succeeds_on_second_attempt(
        self,
        synthetic_desktop: np.ndarray,
        mock_detection_result: DetectionResult,
    ) -> None:
        grounder = VisualGrounder(confidence_threshold=0.25)
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # First attempt fails
            return mock_detection_result  # Second succeeds

        with mock.patch.object(grounder, "locate", side_effect=side_effect):
            result = grounder.locate_with_retry(
                screenshot_fn=lambda: synthetic_desktop,
                target_name="Notepad",
                retries=3,
            )

        assert result is not None
        assert call_count[0] == 2

    def test_locate_with_retry_returns_none_after_all_attempts(
        self, synthetic_desktop: np.ndarray
    ) -> None:
        grounder = VisualGrounder()

        with mock.patch.object(grounder, "locate", return_value=None):
            result = grounder.locate_with_retry(
                screenshot_fn=lambda: synthetic_desktop,
                target_name="Notepad",
                retries=3,
            )
        assert result is None

    def test_confidence_threshold_restored_after_retry(
        self, synthetic_desktop: np.ndarray
    ) -> None:
        grounder = VisualGrounder(confidence_threshold=0.5)
        original = grounder._ranker.confidence_threshold

        with mock.patch.object(grounder, "locate", return_value=None):
            grounder.locate_with_retry(
                screenshot_fn=lambda: synthetic_desktop,
                target_name="Notepad",
                retries=2,
            )

        # Threshold should be restored regardless of retry outcome
        assert grounder._ranker.confidence_threshold == pytest.approx(original)
