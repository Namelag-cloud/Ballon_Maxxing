from __future__ import annotations

import os

from game_profiles.base import GameProfile, GameType, GameAction
from game_profiles.cookie_clicker import CookieClickerProfile
from game_profiles.balloon_game import BalloonGameProfile
from game_profiles.ml_balloon_game import MLBalloonGameProfile

GAME_PROFILES = {
    "cookie": CookieClickerProfile,
    "balloons": BalloonGameProfile,
}


def get_profile(name: str) -> GameProfile:
    cls = GAME_PROFILES.get(name)
    if cls is None:
        raise ValueError(f"Unknown game: {name}. Available: {list(GAME_PROFILES.keys())}")

    if name == "balloons" and os.getenv("GOOGLE_MAX_ML_VISION", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        threshold = float(os.getenv("GOOGLE_MAX_ML_THRESHOLD", "0.70"))
        batch_size = int(os.getenv("GOOGLE_MAX_ML_BATCH_SIZE", "16"))
        checkpoint = os.getenv(
            "GOOGLE_MAX_ML_CHECKPOINT",
            "training/artifacts/mobilenet_v3_small.pt",
        )
        return MLBalloonGameProfile(
            checkpoint=checkpoint,
            threshold=threshold,
            batch_size=batch_size,
        )

    return cls()
