from game_profiles.base import GameProfile, GameType, GameAction
from game_profiles.cookie_clicker import CookieClickerProfile
from game_profiles.balloon_game import BalloonGameProfile

GAME_PROFILES = {
    "cookie": CookieClickerProfile,
    "balloons": BalloonGameProfile,
}


def get_profile(name: str) -> GameProfile:
    cls = GAME_PROFILES.get(name)
    if cls is None:
        raise ValueError(f"Unknown game: {name}. Available: {list(GAME_PROFILES.keys())}")
    return cls()
