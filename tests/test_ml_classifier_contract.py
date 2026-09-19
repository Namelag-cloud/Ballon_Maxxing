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


def test_ml_profile_uses_spatial_boundary_not_parent_semantic_detect() -> None:
    import inspect

    source = inspect.getsource(MLBalloonGameProfile.detect)
    assert "super().detect(" not in source
    assert "detect_spatial_candidates(" in source
    assert "_detect_super_g_group(" not in source
    assert "_classify_golden_duos(" not in source
    assert "_g_detector" not in source


def test_ml_profile_has_no_legacy_semantic_lane() -> None:
    import inspect

    source = inspect.getsource(MLBalloonGameProfile.detect)
    assert "_detect_multicolor_special" not in source
    assert "_detect_super_g_group" not in source
    assert "_classify_golden_duos" not in source
    assert "_g_detector" not in source
    assert "semantic_source" in source
