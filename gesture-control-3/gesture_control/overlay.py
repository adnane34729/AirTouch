"""On-screen display.

The original drew one line of white text and a hand-written two-line legend, both
without any backing, so on a bright background the text was unreadable and the
legend could disagree with the recogniser.

This draws a layered HUD:

* **info panel** -- frame rate, resolution, source, detector, inference time;
* **status panel** -- paused/dry-run badges and the most recent action outcome,
  including failures, so a broken gesture is visible instead of mysterious;
* **gesture guide** -- generated from the same specs the recogniser uses, with
  the currently matched row highlighted;
* **hand state panel** -- per-finger curl/reach numbers with an up/down
  indicator, the arming progress bar (which explains *why* nothing has fired
  yet), and live system levels;
* **hand skeleton** -- per-finger colouring so an unexpectedly "up" finger is
  obvious at a glance.

Everything scales with the frame size and degrades gracefully: panels are only
drawn when they fit, so a small window still gets a usable readout rather than
overlapping text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .config import OverlayConfig
from .gestures import GestureSet, GestureSpec
from .icons import IconLibrary
from .types import (
    FINGER_CONNECTIONS,
    FINGER_JOINTS,
    FINGERS,
    PALM_CONNECTIONS,
    WRIST,
    Finger,
    HandPose,
)

__all__ = ["HudState", "HudRenderer", "PALETTE"]

#: BGR colours. Kept in one place so panels agree with each other.
PALETTE: Mapping[str, tuple[int, int, int]] = {
    "text": (238, 238, 238),
    "muted": (150, 150, 155),
    "dim": (95, 95, 100),
    "panel": (18, 18, 22),
    "border": (70, 70, 78),
    "accent": (255, 190, 60),
    "ok": (110, 220, 110),
    "warn": (60, 180, 240),
    "error": (80, 80, 235),
    "up": (110, 220, 110),
    "down": (110, 110, 118),
    "thumb": (90, 200, 255),
    "index": (110, 230, 130),
    "middle": (255, 200, 80),
    "ring": (230, 130, 230),
    "pinky": (130, 170, 255),
}

_FINGER_COLOURS: Mapping[Finger, tuple[int, int, int]] = {
    Finger.THUMB: PALETTE["thumb"],
    Finger.INDEX: PALETTE["index"],
    Finger.MIDDLE: PALETTE["middle"],
    Finger.RING: PALETTE["ring"],
    Finger.PINKY: PALETTE["pinky"],
}


@dataclass
class HudState:
    """Everything the HUD needs for one frame.

    Deliberately a plain data holder: the renderer has no references to the
    detector, the registry or the debouncer, which keeps it trivially testable
    and makes it impossible for drawing to have side effects on control state.
    """

    fps: float = 0.0
    frame_index: int = 0
    resolution: tuple[int, int] = (0, 0)
    source: str = ""
    detector: str = ""
    inference_ms: float = 0.0
    warmup_frames: int = 0

    pose: HandPose | None = None
    """The primary hand, drawn as the skeleton."""

    hands: tuple[HandPose, ...] = ()
    """Every visible hand, for the two-hand indicator."""

    candidate: GestureSpec | None = None
    active: GestureSpec | None = None
    armed_fraction: float = 0.0
    held_seconds: float = 0.0
    repeats: int = 0

    two_hand_candidate: object | None = None
    two_hand_active: object | None = None
    two_hand_armed_fraction: float = 0.0

    last_action: str = ""
    last_message: str = ""
    last_ok: bool = True
    last_action_age: float = 0.0

    banner: str = ""
    banner_age: float = 0.0
    banner_ok: bool = True
    banner_duration: float = 1.2

    mode_announce: str = ""
    mode_announce_age: float = 99.0
    mode_announce_duration: float = 1.8

    paused: bool = False
    dry_run: bool = False
    show_hand: bool = True

    volume: float | None = None
    brightness: float | None = None

    warnings: Sequence[str] = field(default_factory=tuple)
    backend_notes: Sequence[str] = field(default_factory=tuple)

    gesture_set: GestureSet | None = None
    """Gestures of the *current mode*, so the guide always matches the mode."""

    action_labels: Mapping[str, str] = field(default_factory=dict)
    """Action id -> plain-language name, for the gesture legend.

    The legend names what a gesture *does* ("Volume +"), not what it is called
    internally ("volume_up"), because the point of the legend is that the user
    does not have to learn the naming scheme.
    """

    mode: object | None = None
    mode_index: int = 0
    mode_count: int = 1
    modes: Sequence[object] = field(default_factory=tuple)
    two_hand_set: object | None = None


class HudRenderer:
    """Draws :class:`HudState` onto a frame."""

    #: Text scale for legend rows, shared by measurement and drawing so a row
    #: is always exactly as wide as it was planned to be.
    _LEGEND_TEXT = 0.46

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self.config = config or OverlayConfig()
        # Shape pictograms are cached: the legend redraws every frame but a given
        # hand shape never changes.
        self._icons = IconLibrary()

    # -- cv2 access -------------------------------------------------------- #

    @staticmethod
    def _cv2():
        import cv2

        return cv2

    def _scale_for(self, frame: np.ndarray) -> float:
        """Font/spacing multiplier derived from frame size and config."""
        height, width = frame.shape[:2]
        base = min(width / 1280.0, height / 720.0)
        return float(np.clip(base * self.config.scale, 0.45, 2.5))

    # -- primitives -------------------------------------------------------- #

    def _panel(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        alpha: float = 0.62,
        border: bool = True,
    ) -> None:
        """Translucent panel with a hairline border.

        Drawn by blending a copy of the region, so the video stays visible
        underneath while text keeps enough contrast to read on any background --
        which is precisely what a bare ``putText`` fails to guarantee.
        """
        cv2 = self._cv2()
        frame_height, frame_width = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame_width, x + width), min(frame_height, y + height)
        if x1 <= x0 or y1 <= y0:
            return
        region = frame[y0:y1, x0:x1]
        overlay = np.empty_like(region)
        overlay[:] = PALETTE["panel"]
        cv2.addWeighted(overlay, alpha, region, 1.0 - alpha, 0.0, region)
        if border:
            cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), PALETTE["border"], 1, cv2.LINE_AA)

    def _text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        *,
        scale: float = 0.5,
        colour: tuple[int, int, int] | None = None,
        thickness: int = 1,
        shadow: bool = True,
    ) -> None:
        cv2 = self._cv2()
        if not text:
            return
        colour = colour or PALETTE["text"]
        if shadow:
            cv2.putText(
                frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
            )
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, colour, thickness, cv2.LINE_AA,
        )

    def _bar(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        fraction: float,
        colour: tuple[int, int, int],
        *,
        track: bool = True,
    ) -> None:
        cv2 = self._cv2()
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if track:
            cv2.rectangle(frame, (x, y), (x + width, y + height), PALETTE["dim"], -1)
        filled = int(round(width * fraction))
        if filled > 0:
            cv2.rectangle(frame, (x, y), (x + filled, y + height), colour, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), PALETTE["border"], 1, cv2.LINE_AA)

    def _dot(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        radius: int,
        colour: tuple[int, int, int],
        *,
        filled: bool,
    ) -> None:
        cv2 = self._cv2()
        if filled:
            cv2.circle(frame, (x, y), radius, colour, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), radius, colour, 1, cv2.LINE_AA)

    # -- hand skeleton ----------------------------------------------------- #

    def draw_hand(self, frame: np.ndarray, pose: HandPose, *, primary: bool = True) -> None:
        """Draw one hand, colouring each finger by its own state.

        Uses the raw normalised landmarks against the frame size, so it lines up
        with the video exactly. The secondary hand is drawn thinner and without
        tip labels, so the primary hand stays readable when both are in frame.
        """
        cv2 = self._cv2()
        height, width = frame.shape[:2]
        raw = pose.observation.raw
        points = [
            (int(round(float(raw[index, 0]) * width)), int(round(float(raw[index, 1]) * height)))
            for index in range(len(raw))
        ]
        thickness = 3 if primary else 2

        for start, end in PALM_CONNECTIONS:
            cv2.line(frame, points[start], points[end], PALETTE["dim"], thickness, cv2.LINE_AA)

        for finger in FINGERS:
            reading = pose.readings.get(finger)
            up = bool(reading and reading.extended)
            colour = _FINGER_COLOURS[finger] if up else PALETTE["down"]
            if not primary:
                colour = tuple(int(channel * 0.7) for channel in colour)
            for start, end in FINGER_CONNECTIONS[finger]:
                cv2.line(frame, points[start], points[end], colour, thickness, cv2.LINE_AA)

        for finger in FINGERS:
            reading = pose.readings.get(finger)
            up = bool(reading and reading.extended)
            colour = _FINGER_COLOURS[finger] if up else PALETTE["down"]
            if not primary:
                colour = tuple(int(channel * 0.7) for channel in colour)
            joints = FINGER_JOINTS[finger]
            for position, index in enumerate(joints):
                radius = (6 if primary else 4) if position == len(joints) - 1 else 4
                self._dot(frame, *points[index], radius, colour, filled=up or position == 0)

        # Wrist marker, so the palm origin is unambiguous.
        cv2.circle(frame, points[WRIST], 7 if primary else 5, PALETTE["accent"], 2, cv2.LINE_AA)

        if self.config.show_finger_states and primary:
            for finger in FINGERS:
                reading = pose.readings.get(finger)
                up = bool(reading and reading.extended)
                x, y = points[FINGER_JOINTS[finger][-1]]
                self._text(
                    frame, finger.initial, x + 8, y - 6, scale=0.45,
                    colour=_FINGER_COLOURS[finger] if up else PALETTE["muted"],
                    thickness=2 if up else 1,
                )

    # -- panels ------------------------------------------------------------ #

    def _mode_badge(self, frame: np.ndarray, state: HudState, s: float, margin: int) -> int:
        """Top-centre badge naming the active mode.

        Placed centrally and drawn largest of the panels, because "which mode am
        I in" is the single question that decides what every gesture will do. A
        mode indicator tucked into a corner would be worse than useless.
        """
        cv2 = self._cv2()
        frame_height, frame_width = frame.shape[:2]
        if state.mode is None:
            return 0

        label = getattr(state.mode, "label", "?")
        accent_key = getattr(state.mode, "accent", "accent")
        accent = PALETTE.get(accent_key, PALETTE["accent"])
        counter = f"{state.mode_index + 1}/{state.mode_count}"

        big = 0.85 * s
        small = 0.45 * s
        (big_width, big_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, big, 3)
        (small_width, small_height), _ = cv2.getTextSize(
            f"MODE {counter}", cv2.FONT_HERSHEY_SIMPLEX, small, 1
        )
        content_width = max(big_width, small_width)
        padding = int(16 * s)
        width = content_width + 2 * padding
        height = big_height + small_height + int(34 * s)
        x = (frame_width - width) // 2
        y = margin

        self._panel(frame, x, y, width, height, alpha=0.72)
        # An accent bar makes the current mode identifiable at a glance, and
        # distinguishable between modes without reading the label.
        cv2.rectangle(
            frame, (x, y), (x + width, y + max(2, int(4 * s))), accent, -1
        )
        self._text(
            frame, f"MODE {counter}", x + padding, y + int(20 * s),
            scale=small, colour=PALETTE["muted"], thickness=1,
        )
        self._text(
            frame, label, x + padding, y + int(20 * s) + big_height + int(4 * s),
            scale=big, colour=accent, thickness=3,
        )
        return height

    def _mode_strip(self, frame: np.ndarray, state: HudState, s: float, margin: int, top: int) -> int:
        """A row of mode names with the active one highlighted."""
        if state.mode_count <= 1 or not state.modes:
            return 0
        cv2 = self._cv2()
        frame_height, frame_width = frame.shape[:2]
        names = [getattr(mode, "label", "?") for mode in state.modes]
        scale = 0.42 * s
        widths = [
            cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for name in names
        ]
        padding = int(8 * s)
        total = sum(widths) + padding * (len(names) + 1)
        x = (frame_width - total) // 2
        if x < margin:
            return 0
        height = int(24 * s)
        self._panel(frame, x, top, total, height, alpha=0.55)

        cursor = x + padding
        for index, (name, name_width) in enumerate(zip(names, widths)):
            current = index == state.mode_index
            self._text(
                frame,
                name,
                cursor,
                top + int(17 * s),
                scale=scale,
                colour=PALETTE["text"] if current else PALETTE["dim"],
                thickness=2 if current else 1,
            )
            if current:
                cv2.line(
                    frame,
                    (cursor, top + height - int(3 * s)),
                    (cursor + name_width, top + height - int(3 * s)),
                    PALETTE["accent"],
                    max(1, int(2 * s)),
                    cv2.LINE_AA,
                )
            cursor += name_width + padding
        return height

    def _info_panel(self, frame: np.ndarray, state: HudState, s: float, margin: int) -> int:
        """Top-left panel. Returns its height in pixels."""
        cv2 = self._cv2()
        line_height = int(round(20 * s))
        lines = [
            ("HAND GESTURE CONTROL", PALETTE["accent"], 0.56, 2),
            (f"{state.fps:5.1f} fps   {state.resolution[0]}x{state.resolution[1]}", PALETTE["text"], 0.46, 1),
        ]
        if self.config.show_timings:
            lines.append(
                (f"detect {state.inference_ms:5.1f} ms   frame {state.frame_index}", PALETTE["muted"], 0.42, 1)
            )
        if state.source:
            lines.append((f"src  {state.source}", PALETTE["muted"], 0.42, 1))
        if state.detector:
            lines.append((f"det  {state.detector}", PALETTE["muted"], 0.42, 1))

        width = int(round(max(len(text) for text, *_ in lines) * 9.5 * s)) + int(24 * s)
        height = line_height * len(lines) + int(16 * s)
        self._panel(frame, margin, margin, width, height)

        y = margin + int(20 * s)
        for text, colour, scale, thickness in lines:
            self._text(frame, text, margin + int(12 * s), y, scale=scale * s, colour=colour, thickness=thickness)
            y += line_height
        return height

    def _status_panel(self, frame: np.ndarray, state: HudState, s: float, margin: int) -> None:
        """Top-right panel: badges and the last action outcome."""
        cv2 = self._cv2()
        frame_height, frame_width = frame.shape[:2]
        line_height = int(round(20 * s))

        lines: list[tuple[str, tuple[int, int, int], float, int]] = []
        if state.paused:
            lines.append(("|| PAUSED  (press p to resume)", PALETTE["warn"], 0.5, 2))
        if state.dry_run:
            lines.append(("DRY RUN - no system changes", PALETTE["warn"], 0.5, 2))

        if state.last_action:
            colour = PALETTE["ok"] if state.last_ok else PALETTE["error"]
            age = f"{state.last_action_age:.1f}s ago" if state.last_action_age else "now"
            lines.append((f"{state.last_action}  ({age})", colour, 0.48, 2))
            if state.last_message:
                lines.append((state.last_message[:58], PALETTE["text"], 0.42, 1))
        else:
            lines.append(("no action yet", PALETTE["muted"], 0.46, 1))

        for warning in list(state.warnings)[:2]:
            lines.append((warning[:58], PALETTE["warn"], 0.4, 1))

        width = int(round(max(len(text) for text, *_ in lines) * 9.5 * s)) + int(24 * s)
        height = line_height * len(lines) + int(16 * s)
        x = frame_width - width - margin
        if x < margin:
            return
        self._panel(frame, x, margin, width, height)

        y = margin + int(20 * s)
        for text, colour, scale, thickness in lines:
            self._text(frame, text, x + int(12 * s), y, scale=scale * s, colour=colour, thickness=thickness)
            y += line_height

    # -- gesture legend ---------------------------------------------------- #

    def _entry_width(self, text: str, icon_height: int, s: float, *, icons: int = 1) -> int:
        """Pixel width of one legend entry: icons, gap, then its text."""
        cv2 = self._cv2()
        text_width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self._LEGEND_TEXT * s, 1
        )[0][0]
        icon_span = icons * icon_height + max(0, icons - 1) * int(round(4 * s))
        return icon_span + int(round(7 * s)) + text_width

    def _plan_grid(
        self,
        entries: list[tuple[str, str]],
        *,
        s: float,
        icons: int,
        available_height: int,
        max_columns: int = 3,
    ) -> tuple[int, int, int, int]:
        """Choose icon size, column count and panel width for a set of entries.

        A legend that overflows the frame is worse than a small one, so the icons
        shrink and then the column count grows until the table fits. Returns
        ``(icon_height, columns, row_height, width)``.
        """
        gap = int(round(12 * s))
        padding = int(round(12 * s))
        header = int(round(20 * s))
        for icon_height in (
            int(round(38 * s)),
            int(round(33 * s)),
            int(round(28 * s)),
            int(round(22 * s)),
            int(round(17 * s)),
            # Last resort: a cramped legend on a small window still tells you the
            # shape-to-action mapping, which is the entire point of having one.
            int(round(13 * s)),
        ):
            row_height = icon_height + int(round(7 * s))
            for columns in range(1, max_columns + 1):
                chunk = max(1, -(-len(entries) // columns))
                rows = chunk
                if padding + header + rows * row_height + padding > available_height:
                    continue
                widths = [
                    max(
                        (
                            self._entry_width(text, icon_height, s, icons=icons)
                            for _, text in entries[column * chunk : (column + 1) * chunk]
                        ),
                        default=0,
                    )
                    for column in range(columns)
                ]
                return (
                    icon_height,
                    columns,
                    row_height,
                    sum(widths) + gap * (columns - 1) + 2 * padding,
                )
        # Nothing fits: the caller drops the panel rather than drawing off-screen.
        return 0, 0, 0, 0

    def _guide_panel(
        self,
        frame: np.ndarray,
        state: HudState,
        s: float,
        margin: int,
        *,
        top_limit: int = 0,
    ) -> int:
        """Bottom-left panel: what every gesture in the current mode does.

        Rendered as a *picture* of the hand shape beside the plain name of the
        action, so the table can be read without knowing which column is which
        finger or what any symbol means. It is built from the active mode's specs
        and its action labels, so switching mode rewrites the table -- which is
        what a mode system needs in order to be usable at all.
        """
        specs = list(state.gesture_set) if state.gesture_set is not None else []
        if not specs:
            return 0

        cv2 = self._cv2()
        frame_height, frame_width = frame.shape[:2]
        entries = [(spec.name, self._action_label(state, spec.action)) for spec in specs]
        available = frame_height - 2 * margin - top_limit
        if available < int(70 * s):
            return 0

        icon_height, columns, row_height, width = self._plan_grid(
            entries, s=s, icons=1, available_height=available
        )
        if width <= 0:
            return 0

        padding = int(round(12 * s))
        header_height = int(round(20 * s))
        chunk = max(1, -(-len(entries) // columns))
        height = padding + header_height + chunk * row_height + padding
        y0 = frame_height - height - margin
        if y0 < margin + top_limit:
            return 0

        self._panel(frame, margin, y0, width, height)
        mode_label = getattr(state.mode, "label", "GESTURES")
        self._text(
            frame,
            f"GESTURES - {mode_label} MODE",
            margin + padding,
            y0 + padding + int(round(13 * s)),
            scale=0.5 * s,
            colour=PALETTE["accent"],
            thickness=2,
        )
        y = y0 + padding + header_height

        # Column positions come from the measured width of each column, so the
        # second column does not collide with a long entry in the first.
        gap = int(round(12 * s))
        column_x: list[int] = []
        cursor = margin + padding
        for column in range(columns):
            column_x.append(cursor)
            widest = max(
                (
                    self._entry_width(text, icon_height, s)
                    for _, text in entries[column * chunk : (column + 1) * chunk]
                ),
                default=0,
            )
            cursor += widest + gap

        for index, (name, text) in enumerate(entries):
            column = index // chunk
            row = index - column * chunk
            is_active = state.active is not None and state.active.name == name
            is_candidate = state.candidate is not None and state.candidate.name == name
            self._draw_entry(
                frame,
                column_x[column],
                y + row * row_height,
                icon_height,
                [specs[index].requirements],
                text,
                s,
                highlight=is_active,
                pending=is_candidate,
            )
        return height

    def _two_hand_panel(
        self,
        frame: np.ndarray,
        state: HudState,
        s: float,
        top: int,
        *,
        margin: int,
    ) -> int:
        """Top-centre panel: the two-hand gestures, beside the mode they change.

        Kept away from the mode's own gesture table on purpose. The two-hand
        gestures are global -- they do not change with the mode -- and they answer
        "how do I change this / get out", so they belong next to the mode
        indicator rather than buried under ten mode-specific rows.
        """
        gestures = (
            list(getattr(state.two_hand_set, "gestures", ()))
            if state.two_hand_set is not None
            else []
        )
        if not gestures:
            return 0
        cv2 = self._cv2()
        frame_height, frame_width = frame.shape[:2]
        entries = [(g.name, self._action_label(state, g.action)) for g in gestures]
        available = frame_height - top - 2 * margin
        icon_height, columns, row_height, width = self._plan_grid(
            entries, s=s, icons=2, available_height=available, max_columns=2
        )
        if width <= 0:
            return 0

        padding = int(round(12 * s))
        header_height = int(round(18 * s))
        chunk = max(1, -(-len(entries) // columns))
        height = padding + header_height + chunk * row_height + padding
        x0 = (frame_width - width) // 2
        if x0 < margin or top + height > frame_height - margin:
            # Too wide for the gap between the corner panels: skip it rather than
            # overlap them.
            return 0

        self._panel(frame, x0, top, width, height)
        self._text(
            frame,
            "TWO HANDS - change mode, pause, quit",
            x0 + padding,
            top + padding + int(round(12 * s)),
            scale=0.46 * s,
            colour=PALETTE["warn"],
            thickness=2,
        )
        y = top + padding + header_height
        gap = int(round(14 * s))
        column_x: list[int] = []
        cursor = x0 + padding
        for column in range(columns):
            column_x.append(cursor)
            widest = max(
                (
                    self._entry_width(text, icon_height, s, icons=2)
                    for _, text in entries[column * chunk : (column + 1) * chunk]
                ),
                default=0,
            )
            cursor += widest + gap

        for index, (name, text) in enumerate(entries):
            column = index // chunk
            row = index - column * chunk
            gesture = gestures[index]
            left = gesture.shape if gesture.symmetric else (gesture.left or {})
            right = gesture.shape if gesture.symmetric else (gesture.right or {})
            is_active = (
                state.two_hand_active is not None
                and getattr(state.two_hand_active, "name", None) == name
            )
            is_candidate = (
                state.two_hand_candidate is not None
                and getattr(state.two_hand_candidate, "name", None) == name
            )
            self._draw_entry(
                frame,
                column_x[column],
                y + row * row_height,
                icon_height,
                [left or {}, right or {}],
                text,
                s,
                highlight=is_active,
                pending=is_candidate,
                accent=PALETTE["warn"],
                dim=not (is_active or is_candidate),
            )
        return height

    def _action_label(self, state: HudState, action_id: str) -> str:
        """Plain-language name for an action, falling back to its identifier."""
        known = state.action_labels.get(action_id)
        return known or action_id.replace("_", " ").capitalize()

    def _draw_entry(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        icon_height: int,
        shapes: Sequence[object],
        text: str,
        s: float,
        *,
        highlight: bool,
        pending: bool,
        accent: tuple[int, int, int] | None = None,
        dim: bool = False,
    ) -> None:
        """One legend row: one or two hand pictograms, then the action name."""
        cv2 = self._cv2()
        gap = int(round(6 * s))
        if highlight or pending:
            marker = PALETTE["ok"] if highlight else PALETTE["accent"]
            width = self._entry_width(text, icon_height, s, icons=len(shapes))
            cv2.rectangle(
                frame,
                (x - int(5 * s), y - int(4 * s)),
                (x + width + int(2 * s), y + icon_height + int(2 * s)),
                tuple(int(channel * 0.30) for channel in marker),
                -1,
            )
        cursor = x
        for index, shape in enumerate(shapes):
            self._draw_shape_icon(
                frame, cursor, y, icon_height, shape, s, dim=dim, mirror=index == 1
            )
            cursor += icon_height + (gap if index == 0 else 0)
        colour = PALETTE["ok"] if highlight else (
            PALETTE["accent"] if pending else (accent or PALETTE["text"])
        )
        self._text(
            frame,
            text,
            cursor,
            y + icon_height - int(round(7 * s)),
            scale=self._LEGEND_TEXT * s,
            colour=colour,
            thickness=2 if (highlight or pending) else 1,
        )

    def _draw_shape_icon(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        height: int,
        requirements,
        s: float,
        *,
        dim: bool = False,
        mirror: bool = False,
    ) -> int:
        """Draw a miniature hand showing a gesture's shape. Returns its width.

        Extended fingers are drawn in their own colour, curled fingers in the
        neutral "down" colour, and fingers the gesture ignores in the dimmest
        colour at half curl -- so "either way is fine" is visible at a glance
        rather than being one more symbol to learn.
        """
        cv2 = self._cv2()
        icon = self._icons.get(requirements)
        width = max(6, int(round(height * icon.aspect)))
        if mirror:
            # Reflect about the icon's box, so the pair in a two-hand row reads
            # as a left and a right hand rather than one shape drawn twice.
            points = [
                (int(round(x + (1.0 - nx) * width)), int(round(y + ny * height)))
                for nx, ny in icon.points
            ]
        else:
            points = [
                (int(round(x + nx * width)), int(round(y + ny * height)))
                for nx, ny in icon.points
            ]

        def fade(colour: tuple[int, int, int]) -> tuple[int, int, int]:
            return tuple(int(channel * 0.55) for channel in colour) if dim else colour

        thickness = 2 if height >= 20 else 1
        for start, end in PALM_CONNECTIONS:
            cv2.line(
                frame, points[start], points[end], fade(PALETTE["dim"]), thickness, cv2.LINE_AA
            )

        for finger, bones in icon.finger_bones():
            position = FINGERS.index(finger)
            if icon.unspecified[position]:
                colour = PALETTE["dim"]
            elif icon.extended[position]:
                colour = _FINGER_COLOURS[finger]
            else:
                colour = PALETTE["down"]
            for start, end in bones:
                cv2.line(frame, points[start], points[end], fade(colour), thickness, cv2.LINE_AA)
            tip = FINGER_JOINTS[finger][-1]
            radius = max(1, int(round(height * 0.075)))
            cv2.circle(frame, points[tip], radius, fade(colour), -1, cv2.LINE_AA)

        # Wrist marker, so the orientation of the little hand is obvious.
        cv2.circle(
            frame,
            points[WRIST],
            max(2, int(round(height * 0.11))),
            fade(PALETTE["accent"]),
            thickness,
            cv2.LINE_AA,
        )
        return width

    def _state_panel(self, frame: np.ndarray, state: HudState, s: float, margin: int) -> None:
        """Bottom-right panel: finger readouts, arming progress, levels."""
        frame_height, frame_width = frame.shape[:2]
        line_height = int(round(20 * s))
        row_height = int(round(19 * s))

        row_count = len(FINGERS) if state.pose is not None else 1

        header = f"{state.pose.handedness.value} hand" if state.pose is not None else "waiting"
        progress = (
            state.candidate.label
            if state.candidate is not None
            else (state.active.label if state.active else "-")
        )

        levels = []
        if self.config.show_system_levels and state.volume is not None:
            levels.append(("VOL", state.volume, PALETTE["warn"]))
        if self.config.show_system_levels and state.brightness is not None:
            levels.append(("BRT", state.brightness / 100.0, PALETTE["accent"]))

        # Resolved before the height is computed: a two-hand gesture that is
        # arming needs an extra row, and the panel has to reserve it up front.
        two_hand_candidate = state.two_hand_candidate
        two_hand_label = (
            getattr(two_hand_candidate, "label", None) if two_hand_candidate else None
        )

        height = (
            int(16 * s)
            + line_height
            + row_height * row_count
            + int(8 * s)
            + line_height
            + int(8 * s)
            + line_height * len(levels)
            + (line_height + int(10 * s) if two_hand_label else 0)
        )
        width = int(round(33 * 9.5 * s)) + int(24 * s)
        x = frame_width - width - margin
        y0 = frame_height - height - margin
        if x < margin or y0 < margin:
            return
        self._panel(frame, x, y0, width, height)

        y = y0 + int(20 * s)
        self._text(frame, header, x + int(12 * s), y, scale=0.5 * s, colour=PALETTE["accent"], thickness=2)
        y += line_height

        if state.pose is None:
            self._text(
                frame, "no hand detected", x + int(12 * s), y,
                scale=0.42 * s, colour=PALETTE["muted"], thickness=1,
            )
            y += row_height
        else:
            for finger in FINGERS:
                reading = state.pose.readings.get(finger)
                if reading is None:
                    continue
                up = reading.extended
                extra = ""
                if finger is Finger.THUMB and not math.isnan(reading.abduction):
                    extra = f"  abd {reading.abduction:4.2f}"
                row = f"{finger.initial} {'UP  ' if up else 'down'} curl {reading.curl:5.1f}  reach {reading.reach:4.2f}{extra}"
                self._dot(
                    frame,
                    x + int(18 * s),
                    y - int(4 * s),
                    max(2, int(4 * s)),
                    _FINGER_COLOURS[finger] if up else PALETTE["down"],
                    filled=up,
                )
                self._text(
                    frame, row, x + int(30 * s), y, scale=0.42 * s,
                    colour=PALETTE["text"] if up else PALETTE["muted"], thickness=1,
                )
                y += row_height

        y += int(6 * s)
        self._text(frame, f"match {progress}", x + int(12 * s), y, scale=0.45 * s,
                   colour=PALETTE["text"], thickness=1)
        y += int(10 * s)
        self._bar(
            frame, x + int(12 * s), y, width - int(24 * s), max(3, int(6 * s)),
            state.armed_fraction,
            PALETTE["ok"] if state.armed_fraction >= 1.0 else PALETTE["accent"],
        )
        y += line_height

        if two_hand_label:
            self._text(
                frame, f"two hands: {two_hand_label}", x + int(12 * s), y,
                scale=0.45 * s, colour=PALETTE["warn"], thickness=2,
            )
            y += int(10 * s)
            self._bar(
                frame, x + int(12 * s), y, width - int(24 * s), max(3, int(6 * s)),
                state.two_hand_armed_fraction,
                PALETTE["warn"],
            )
            y += line_height

        for label, fraction, colour in levels:
            self._text(frame, label, x + int(12 * s), y + int(10 * s), scale=0.42 * s,
                       colour=PALETTE["muted"], thickness=1)
            self._bar(
                frame, x + int(44 * s), y + int(2 * s), width - int(58 * s), max(3, int(6 * s)),
                fraction, colour,
            )
            y += line_height

    def _banner(
        self, frame: np.ndarray, state: HudState, s: float, *, min_top: int = 0
    ) -> None:
        """Centred banner announcing an action, fading out over its lifetime."""
        cv2 = self._cv2()
        if not state.banner or state.banner_age > state.banner_duration:
            return
        frame_height, frame_width = frame.shape[:2]
        fade = 1.0 - (state.banner_age / max(1e-6, state.banner_duration))
        if fade <= 0.02:
            return

        text = state.banner
        scale = 0.8 * s
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
        )
        x = (frame_width - text_width) // 2
        # Sit in the upper-middle by default, but never on top of the panels that
        # are already there.
        y = max(int(frame_height * 0.20) + text_height, min_top + text_height)
        padding = int(14 * s)
        if y + baseline + padding > frame_height - int(60 * s):
            return

        self._panel(
            frame, x - padding, y - text_height - padding,
            text_width + 2 * padding, text_height + baseline + 2 * padding,
            alpha=0.5 * fade + 0.15,
        )
        colour = PALETTE["ok"] if state.banner_ok else PALETTE["error"]
        blend = tuple(int(channel * (0.45 + 0.55 * fade)) for channel in colour)
        self._text(frame, text, x, y, scale=scale, colour=blend, thickness=2)

    # -- entry point ------------------------------------------------------- #

    def render(self, frame: np.ndarray, state: HudState) -> np.ndarray:
        """Draw the HUD. Returns the frame (modified in place, for speed)."""
        if frame is None or frame.size == 0:
            return frame
        s = self._scale_for(frame)
        margin = int(round(10 * s))

        if self.config.show_skeleton:
            for pose in (state.hands or ((state.pose,) if state.pose else ())):
                if pose is not None:
                    self.draw_hand(frame, pose, primary=pose is state.pose)

        info_height = self._info_panel(frame, state, s, margin)
        self._status_panel(frame, state, s, margin)
        badge_height = self._mode_badge(frame, state, s, margin)
        strip_height = 0
        if badge_height:
            strip_height = self._mode_strip(
                frame, state, s, margin, margin + badge_height + int(6 * s)
            )
        controls_bottom = margin + badge_height + strip_height
        if self.config.show_guide:
            # The two-hand controls sit under the mode indicator: they are the
            # global "how do I change this mode" answer, so they belong with the
            # mode rather than under ten mode-specific rows.
            controls_height = self._two_hand_panel(
                frame,
                state,
                s,
                controls_bottom + int(14 * s),
                margin=margin,
            )
            controls_bottom += int(14 * s) + controls_height
        if self.config.show_guide:
            self._guide_panel(
                frame, state, s, margin, top_limit=info_height + int(10 * s)
            )
        self._state_panel(frame, state, s, margin)
        self._pause_overlay(frame, state, s)
        self._mode_change_overlay(frame, state, s)
        # The action banner would otherwise land on top of the two-hand panel, so
        # it is pushed below whatever the centre column already contains.
        self._banner(frame, state, s, min_top=controls_bottom + int(8 * s))
        return frame

    def _pause_overlay(self, frame: np.ndarray, state: HudState, s: float) -> None:
        """Full-width banner explaining how to get out of the paused state.

        A pause that does not say how to leave it is the trap the original had,
        so the way out is stated on screen rather than left in the docs.
        """
        if not state.paused:
            return
        cv2 = self._cv2()
        frame_height, frame_width = frame.shape[:2]
        # Spell out the way out. Both the two-hand gesture and the key are named,
        # because someone who paused by accident has no idea which one applies.
        text = (
            "PAUSED - gestures ignored.  Resume: shaka shape (thumb + little) on "
            "BOTH hands, or press P"
        )
        scale = 0.6 * s
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
        )
        padding = int(16 * s)
        width = min(frame_width - 2 * int(10 * s), text_width + 2 * padding)
        height = text_height + baseline + 2 * padding
        x = (frame_width - width) // 2
        y = frame_height // 2 - height // 2
        self._panel(frame, x, y, width, height, alpha=0.8)
        cv2.rectangle(
            frame, (x, y), (x + width, y + max(2, int(4 * s))), PALETTE["warn"], -1
        )
        self._text(
            frame, text, x + padding, y + padding + text_height,
            scale=scale, colour=PALETTE["warn"], thickness=2,
        )

    def _mode_change_overlay(self, frame: np.ndarray, state: HudState, s: float) -> None:
        """Big centred announcement when the mode changes."""
        cv2 = self._cv2()
        if not state.mode_announce or state.mode_announce_age > state.mode_announce_duration:
            return
        frame_height, frame_width = frame.shape[:2]
        fade = 1.0 - (state.mode_announce_age / max(1e-6, state.mode_announce_duration))
        if fade <= 0.02:
            return
        accent_key = getattr(state.mode, "accent", "accent")
        accent = PALETTE.get(accent_key, PALETTE["accent"])
        blend = tuple(int(channel * (0.4 + 0.6 * fade)) for channel in accent)

        text = state.mode_announce
        scale = 1.15 * s
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3
        )
        padding = int(20 * s)
        width = text_width + 2 * padding
        height = text_height + baseline + 2 * padding
        x = (frame_width - width) // 2
        y = int(frame_height * 0.34)
        self._panel(frame, x, y, width, height, alpha=0.35 * fade + 0.2)
        self._text(
            frame, text, x + padding, y + padding + text_height,
            scale=scale, colour=blend, thickness=3,
        )
