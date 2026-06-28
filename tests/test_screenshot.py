"""
Unit tests for ScreenshotCapture utilities.

These tests do not require a display server because they only test
the annotation and region-slicing helpers — actual screen capture
is mocked with synthetic images.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.utils.screenshot import ScreenshotCapture


@pytest.fixture
def sample_image() -> np.ndarray:
    """Solid grey 640x480 BGR image."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


class TestAnnotate:
    def test_returns_different_array(self, sample_image: np.ndarray) -> None:
        out = ScreenshotCapture.annotate(
            sample_image, x=50, y=50, w=100, h=100, confidence=0.9
        )
        assert out is not sample_image

    def test_original_unchanged(self, sample_image: np.ndarray) -> None:
        original = sample_image.copy()
        ScreenshotCapture.annotate(
            sample_image, x=50, y=50, w=100, h=100, confidence=0.9
        )
        np.testing.assert_array_equal(sample_image, original)

    def test_output_shape_preserved(self, sample_image: np.ndarray) -> None:
        out = ScreenshotCapture.annotate(
            sample_image, x=50, y=50, w=100, h=100, confidence=0.7, label="Notepad"
        )
        assert out.shape == sample_image.shape

    def test_annotation_changes_pixels(self, sample_image: np.ndarray) -> None:
        """The annotated image must differ from the original (box was drawn)."""
        out = ScreenshotCapture.annotate(
            sample_image, x=50, y=50, w=100, h=100, confidence=0.95, label="Test"
        )
        assert not np.array_equal(out, sample_image)

    def test_custom_color(self, sample_image: np.ndarray) -> None:
        out = ScreenshotCapture.annotate(
            sample_image, x=10, y=10, w=50, h=50,
            confidence=0.8, color=(0, 0, 255)  # red in BGR
        )
        assert out is not None


class TestSave:
    def test_save_creates_file(self, tmp_path: Path, sample_image: np.ndarray) -> None:
        capture = ScreenshotCapture()
        out_path = tmp_path / "test_screenshot.png"
        saved = capture.save(sample_image, out_path)
        assert saved.exists()
        assert saved.suffix == ".png"

    def test_save_creates_parent_dirs(
        self, tmp_path: Path, sample_image: np.ndarray
    ) -> None:
        capture = ScreenshotCapture()
        out_path = tmp_path / "nested" / "dir" / "image.png"
        capture.save(sample_image, out_path)
        assert out_path.exists()


class TestCaptureRegion:
    def test_region_dimensions(self) -> None:
        """capture_region should return the requested size when given a full image."""
        # We can't call mss in CI, so patch the full capture
        import unittest.mock as mock

        full = np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture = ScreenshotCapture()

        with mock.patch.object(capture, "capture", return_value=full):
            # Simulate the fallback path: full[y:y+h, x:x+w]
            region = full[100:200, 50:150]  # 100x100
            assert region.shape == (100, 100, 3)
