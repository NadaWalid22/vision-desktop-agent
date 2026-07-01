"""
Shared pytest fixtures and test configuration.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.grounding.detector import BoundingBox, RegionProposal


@pytest.fixture(scope="session")
def sample_desktop_image() -> np.ndarray:
    """
    960x540 synthetic desktop image used across test modules.
    Contains a bright icon-like patch in the top-left and another in the centre.
    """
    img = np.full((540, 960, 3), 45, dtype=np.uint8)  # Dark grey desktop

    # Icon 1: top-left (simulates Notepad shortcut near corner)
    img[80:128, 60:108] = [30, 160, 220]    # blue icon body
    img[130:145, 55:115] = [220, 220, 220]  # white label text area

    # Icon 2: centre (simulates another desktop icon)
    img[250:298, 460:508] = [60, 200, 80]   # green icon body
    img[300:315, 455:515] = [220, 220, 220]

    return img


@pytest.fixture
def icon_region_proposal() -> RegionProposal:
    """A proposal centred on the top-left icon in sample_desktop_image."""
    return RegionProposal(
        box=BoundingBox(60, 80, 48, 48),
        strategy="contour",
        crop=np.full((48, 48, 3), 140, dtype=np.uint8),
    )
