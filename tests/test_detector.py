"""
Unit tests for the region proposal detector.

These tests are purely CPU-bound and require no GPU, no display,
and no real desktop — they operate on programmatically generated
synthetic images that simulate desktop icon scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.grounding.detector import BoundingBox, RegionProposalDetector


# ─── BoundingBox tests ────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_centre(self) -> None:
        bb = BoundingBox(10, 20, 100, 80)
        assert bb.cx == 60
        assert bb.cy == 60

    def test_area(self) -> None:
        bb = BoundingBox(0, 0, 50, 50)
        assert bb.area == 2500

    def test_x2_y2(self) -> None:
        bb = BoundingBox(5, 10, 40, 30)
        assert bb.x2 == 45
        assert bb.y2 == 40

    def test_iou_identical(self) -> None:
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.iou(bb) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        a = BoundingBox(0, 0, 50, 50)
        b = BoundingBox(100, 100, 50, 50)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial_overlap(self) -> None:
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 100, 100)
        # Overlap: 50x50=2500; union: 10000+10000-2500=17500
        assert a.iou(b) == pytest.approx(2500 / 17500, rel=1e-3)

    def test_as_tuple(self) -> None:
        bb = BoundingBox(1, 2, 3, 4)
        assert bb.as_tuple() == (1, 2, 3, 4)


# ─── RegionProposalDetector tests ────────────────────────────────────────────

@pytest.fixture
def blank_desktop() -> np.ndarray:
    """960x540 grey image simulating an empty desktop."""
    return np.full((540, 960, 3), 50, dtype=np.uint8)


@pytest.fixture
def desktop_with_icon(blank_desktop: np.ndarray) -> np.ndarray:
    """Desktop with a high-contrast 48x48 icon-like region at (200, 150)."""
    img = blank_desktop.copy()
    # Draw a blue square to simulate an icon
    img[150:198, 200:248] = [200, 80, 30]   # BGR
    # White text label below
    img[200:215, 195:260] = [255, 255, 255]
    return img


@pytest.fixture
def detector() -> RegionProposalDetector:
    return RegionProposalDetector(min_size=24, max_size=160, nms_iou=0.5, max_proposals=100)


class TestRegionProposalDetector:
    def test_propose_returns_list(
        self, detector: RegionProposalDetector, blank_desktop: np.ndarray
    ) -> None:
        proposals = detector.propose(blank_desktop)
        assert isinstance(proposals, list)

    def test_proposals_within_bounds(
        self, detector: RegionProposalDetector, blank_desktop: np.ndarray
    ) -> None:
        h, w = blank_desktop.shape[:2]
        for p in detector.propose(blank_desktop):
            assert p.box.x >= 0
            assert p.box.y >= 0
            assert p.box.x2 <= w
            assert p.box.y2 <= h

    def test_proposals_respect_size_limits(
        self, detector: RegionProposalDetector, blank_desktop: np.ndarray
    ) -> None:
        for p in detector.propose(blank_desktop):
            assert detector.min_size <= p.box.w <= detector.max_size
            assert detector.min_size <= p.box.h <= detector.max_size

    def test_proposals_respect_max_count(
        self, detector: RegionProposalDetector, blank_desktop: np.ndarray
    ) -> None:
        proposals = detector.propose(blank_desktop)
        assert len(proposals) <= detector.max_proposals

    def test_crop_shape_matches_box(
        self, detector: RegionProposalDetector, blank_desktop: np.ndarray
    ) -> None:
        proposals = detector.propose(blank_desktop)[:5]
        for p in proposals:
            assert p.crop.shape == (p.box.h, p.box.w, 3)

    def test_icon_region_covered(
        self, detector: RegionProposalDetector, desktop_with_icon: np.ndarray
    ) -> None:
        """At least one proposal should overlap the known icon region."""
        proposals = detector.propose(desktop_with_icon)
        icon_bb = BoundingBox(200, 150, 48, 48)
        overlaps = [p for p in proposals if p.box.iou(icon_bb) > 0.2]
        assert len(overlaps) > 0, "Icon region not covered by any proposal"

    def test_nms_reduces_duplicates(self, detector: RegionProposalDetector) -> None:
        """NMS should remove highly overlapping boxes."""
        # Create two boxes that are 90 % overlapping
        boxes = [
            (BoundingBox(0, 0, 100, 100), "grid"),
            (BoundingBox(5, 5, 100, 100), "grid"),  # Should be suppressed
        ]
        result = detector._nms(boxes)
        assert len(result) == 1

    def test_grid_proposals_cover_image(self) -> None:
        """Grid proposals should span the full image at each scale."""
        det = RegionProposalDetector()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        proposals = det._grid_proposals(img)
        assert len(proposals) > 0
        # All x,y should be non-negative
        for bb, _ in proposals:
            assert bb.x >= 0
            assert bb.y >= 0

    def test_contour_proposals_on_high_contrast(self) -> None:
        """Contour detector should fire on high-contrast icons."""
        det = RegionProposalDetector()
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw a bright square
        img[50:100, 60:110] = 220
        proposals = det._contour_proposals(img)
        assert len(proposals) > 0

    def test_strategy_labels(
        self, detector: RegionProposalDetector, blank_desktop: np.ndarray
    ) -> None:
        proposals = detector.propose(blank_desktop)
        strategies = {p.strategy for p in proposals}
        # At least grid and contour should fire on any image
        assert "grid" in strategies
