from __future__ import annotations

from typing import Any


class DecisionConsole:
    """Compact human-readable view of balloon detection/decision/execution state."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def frame(
        self,
        frame_id: int,
        detections: list[dict[str, Any]],
        decision: Any,
    ) -> None:
        if not self.enabled:
            return

        detected = [self._detection_text(d) for d in detections]
        if detected:
            print(f"[DECISION {frame_id}] DETECTED: " + ", ".join(detected))

        rejected = []
        for d in detections:
            metadata = d.get("metadata", {})
            temporal = metadata.get("temporal", {})
            target_type = d.get("target_type", "unknown")
            target_id = d.get("target_id", "?")
            color = d.get("color_name") or "unknown"
            if temporal.get("static_ui_suspect"):
                rejected.append(
                    f"{color} {target_type}#{target_id} static_ui_suspect"
                )
            elif int(metadata.get("click_attempts", 0)) >= 2:
                rejected.append(
                    f"{color} {target_type}#{target_id} click_attempt_limit"
                )

        for item in rejected:
            print(f"[DECISION {frame_id}] REJECTED: {item}")

        actions = list(getattr(decision, "actions", ()))
        for action in actions:
            self._print_action(frame_id, action)

        g_action = getattr(decision, "g_action", None)
        if g_action is not None:
            self._print_action(frame_id, g_action)

        multicolor = list(getattr(decision, "multicolor_specials", ()))
        executable_multicolor = {
            id(action) for action in actions
        }
        for action in multicolor:
            if id(action) not in executable_multicolor:
                self._print_action(frame_id, action, prefix="OBSERVED")

        notifications = list(getattr(decision, "notifications", ()))
        if notifications:
            print(
                f"[DECISION {frame_id}] NOTIFICATIONS: {len(notifications)}"
            )

        mode = getattr(getattr(decision, "mode", None), "value", None)
        step = getattr(getattr(decision, "special_step", None), "value", None)
        if mode == "special_sequence":
            print(
                f"[DECISION {frame_id}] SUPER-G: required={step or 'unknown'}"
            )

    def execution(self, frame_id: int, execution: Any) -> None:
        if not self.enabled:
            return

        clicks_sent = int(getattr(execution, "clicks_sent", 0))
        outcomes = list(getattr(execution, "outcomes", ()))
        if clicks_sent:
            print(
                f"[EXEC {frame_id}] SENT {clicks_sent} CLICK"
                f"{'' if clicks_sent == 1 else 'S'}"
            )

        for outcome in outcomes:
            target_type = getattr(outcome, "target_type", "unknown")
            target_id = getattr(outcome, "target_id", None)
            status = getattr(outcome, "execution_status", "unknown")
            x = getattr(outcome, "screen_click_x", None)
            y = getattr(outcome, "screen_click_y", None)
            location = f" @ ({x},{y})" if x is not None and y is not None else ""
            ident = f"#{target_id}" if target_id is not None else ""
            print(
                f"[EXEC {frame_id}] {target_type}{ident}"
                f"{location} status={status}"
            )

        completed = getattr(execution, "completed_primary_action", None)
        if completed is not None:
            target_type = getattr(
                getattr(completed, "target_type", None), "value",
                getattr(completed, "target_type", "unknown"),
            )
            color = getattr(
                getattr(completed, "color", None), "value",
                getattr(completed, "color", None),
            )
            print(
                f"[EXEC {frame_id}] COMPLETED: "
                f"{color + ' ' if color else ''}{target_type}"
            )

        if getattr(execution, "twin_completion", None) is not None:
            print(f"[EXEC {frame_id}] TWIN: completed")

        if getattr(execution, "g_click_completed", False):
            print(f"[EXEC {frame_id}] G-SYMBOL: completed")

    @staticmethod
    def _detection_text(detection: dict[str, Any]) -> str:
        color = detection.get("color_name") or "unknown"
        target_type = detection.get("target_type", "unknown")
        target_id = detection.get("target_id", "?")
        center = detection.get("center")
        position = f"@{tuple(center)}" if center else ""
        return f"{color} {target_type}#{target_id}{position}"

    def _print_action(
        self,
        frame_id: int,
        action: Any,
        *,
        prefix: str = "CLICK",
    ) -> None:
        if hasattr(action, "targets"):
            targets = getattr(action, "targets", ())
            pair = ", ".join(self._action_target_text(t) for t in targets)
            print(f"[DECISION {frame_id}] TWIN: {prefix} {pair}")
            return

        target_type = getattr(
            getattr(action, "target_type", None), "value",
            getattr(action, "target_type", "unknown"),
        )
        color = getattr(
            getattr(action, "color", None), "value",
            getattr(action, "color", None),
        )
        target_id = getattr(action, "target_id", None)
        position = getattr(action, "position", None)
        ident = f"#{target_id}" if target_id is not None else ""
        location = f"@{tuple(position)}" if position else ""
        label = f"{color + ' ' if color else ''}{target_type}{ident}{location}"
        print(f"[DECISION {frame_id}] {prefix}: {label}")

    @staticmethod
    def _action_target_text(action: Any) -> str:
        target_type = getattr(
            getattr(action, "target_type", None), "value",
            getattr(action, "target_type", "unknown"),
        )
        color = getattr(
            getattr(action, "color", None), "value",
            getattr(action, "color", None),
        )
        target_id = getattr(action, "target_id", None)
        position = getattr(action, "position", None)
        ident = f"#{target_id}" if target_id is not None else ""
        location = f"@{tuple(position)}" if position else ""
        return f"{color + ' ' if color else ''}{target_type}{ident}{location}"
