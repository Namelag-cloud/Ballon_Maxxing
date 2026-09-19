from __future__ import annotations

from game_profiles.ml_balloon_game import MLBalloonGameProfile
from vision.ml_classifier import DEFAULT_THRESHOLD, IMAGE_SIZE, LABELS


def test_ml_classifier_contract_constants() -> None:
    assert IMAGE_SIZE == 96
    assert DEFAULT_THRESHOLD == 0.70
    assert LABELS == (
        "balloon",
        "duo",
        "notification",
        "super_g_ring",
        "ui",
    )


def test_ml_profile_is_balloon_profile() -> None:
    assert issubclass(MLBalloonGameProfile, object)
