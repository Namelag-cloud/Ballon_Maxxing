from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings
from tracking.tracker import BalloonTrack


@dataclass
class Group:
    tracks: list[BalloonTrack]
    color: str
    has_outline: bool


class GroupingStrategy:
    def __init__(self, settings: Settings) -> None:
        self._config = settings.grouping

    def group(self, tracks: list[BalloonTrack]) -> list[Group]:
        if not self._config.enabled:
            return [Group(tracks=[t], color=t.color, has_outline=t.has_outline) for t in tracks]

        by_color: dict[str, list[BalloonTrack]] = {}
        for t in tracks:
            by_color.setdefault(t.color, []).append(t)

        groups: list[Group] = []
        for color, color_tracks in by_color.items():
            if self._config.same_color_only:
                if len(color_tracks) == 1:
                    groups.append(Group(
                        tracks=color_tracks,
                        color=color,
                        has_outline=color_tracks[0].has_outline,
                    ))
                else:
                    sorted_tracks = self._sort_by_proximity(color_tracks)
                    current_group: list[BalloonTrack] = [sorted_tracks[0]]
                    for t in sorted_tracks[1:]:
                        prev = current_group[-1]
                        dx = prev.position[0] - t.position[0]
                        dy = prev.position[1] - t.position[1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist <= self._config.max_pair_distance:
                            current_group.append(t)
                        else:
                            groups.append(Group(
                                tracks=current_group,
                                color=color,
                                has_outline=any(tt.has_outline for tt in current_group),
                            ))
                            current_group = [t]
                    groups.append(Group(
                        tracks=current_group,
                        color=color,
                        has_outline=any(tt.has_outline for tt in current_group),
                    ))
            else:
                groups.append(Group(
                    tracks=color_tracks,
                    color=color,
                    has_outline=any(t.has_outline for t in color_tracks),
                ))

        return groups

    def _sort_by_proximity(self, tracks: list[BalloonTrack]) -> list[BalloonTrack]:
        if len(tracks) <= 1:
            return list(tracks)
        centroid_x = sum(t.position[0] for t in tracks) / len(tracks)
        centroid_y = sum(t.position[1] for t in tracks) / len(tracks)
        return sorted(tracks, key=lambda t: (
            (t.position[0] - centroid_x) ** 2 + (t.position[1] - centroid_y) ** 2
        ))
