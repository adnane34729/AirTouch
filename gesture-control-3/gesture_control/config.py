"""Configuration: dataclasses, presets, TOML loading, and validation.

Every tunable that used to be a literal buried in the main loop now has a name,
a default, and a documented range. Three things this buys:

* **Typo safety.** An unknown key is a hard error listing the valid keys, rather
  than being silently ignored. A misspelled ``repit_interval`` did nothing in a
  dict-based config; here it fails loudly at startup.
* **Presets.** ``classic`` reproduces the original script's gesture mapping
  exactly, so nothing about existing muscle memory breaks. ``media`` and
  ``minimal`` are alternatives. Presets are plain data, so the shipped preset
  and a user's own config take the same code path -- they cannot drift apart.
* **Round-tripping.** ``--print-config`` emits valid TOML that loads back to an
  identical config (asserted in the tests), so the printed config is a genuinely
  usable starting point rather than a debugging aid.

Precedence, lowest to highest: built-in defaults, then ``--preset``, then the
config file, then explicit command-line flags.
"""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gestures import GestureError, GestureSet, GestureSpec, parse_requirements
from .modes import Mode, ModeError, ModeSet, TwoHandGesture, TwoHandGestureSet
from .pose import FingerThresholds, Thresholds, ThumbThresholds
from .vocabulary import MODE_PRESETS, shape_table, two_hand_basic, two_hand_pro

__all__ = [
    "CameraConfig",
    "DetectorConfig",
    "SmoothingConfig",
    "TrackingConfig",
    "SwitchingConfig",
    "ThresholdConfig",
    "ActionsConfig",
    "OverlayConfig",
    "AppConfig",
    "Preset",
    "PRESETS",
    "ConfigError",
    "load_config",
    "default_config",
    "config_to_toml",
    "available_presets",
    "build_gestures",
    "build_modes",
    "build_two_hand_gestures",
]


class ConfigError(ValueError):
    """Raised when configuration is malformed or contradictory."""


_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class CameraConfig:
    """Webcam capture settings."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    """Mirror the frame horizontally, so moving right moves the cursor right on
    screen. Also what makes MediaPipe's handedness labels correct."""

    fourcc: str = ""
    """Optional pixel format request, e.g. ``"MJPG"``. Blank leaves it alone."""

    reopen_attempts: int = 5
    reopen_delay: float = 1.0
    """If the camera drops out mid-session the app reopens it with this delay
    between attempts. The original simply exited the loop."""


@dataclass
class SmoothingConfig:
    """One Euro landmark filter settings (see :mod:`gesture_control.smoothing`)."""

    enabled: bool = True
    min_cutoff: float = 1.1
    """Hz. Lower = smoother at rest, laggier. 0.5-3.0 is the useful range."""

    beta: float = 0.035
    """Velocity coupling. 0 disables adaptivity (a plain EMA)."""

    d_cutoff: float = 1.0


@dataclass
class DetectorConfig:
    """Hand detector settings."""

    max_hands: int = 2
    """Two by default, because the two-hand gestures (mode switching, pause,
    quit) need both hands visible. Set to 1 to save a little CPU if you only use
    single-hand gestures."""
    model_complexity: int = 1
    """MediaPipe model: 0 fastest, 1 balanced. The original used the default 1."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    stride: int = 1
    """Run detection every Nth frame and reuse the last result in between.

    This is the effective CPU lever. Measured on a typical CPU, inference costs
    ~20-25 ms and does *not* get cheaper with a smaller input, so halving the
    number of inferences is what actually buys headroom. 1 matches the original
    behaviour; higher costs responsiveness while a gesture settles."""

    downscale_width: int = 0
    """If > 0, detection runs on a frame scaled to this width.

    Rarely worth setting. MediaPipe resizes to a fixed 224x224 internally, so
    measured inference time at 480x270 and 1280x720 was within noise of each
    other (~21 ms vs ~20 ms). It only saves the colour conversion and memory
    traffic, so prefer :attr:`stride` unless CPU profiling points here."""


@dataclass
class TrackingConfig:
    """Temporal gesture logic (see :mod:`gesture_control.tracking`)."""

    arm_seconds: float = 0.10
    release_seconds: float = 0.12
    repeat_initial_delay: float = 0.45
    repeat_interval: float = 0.12
    default_cooldown: float = 0.25

    two_hand_arm_seconds: float = 0.55
    """How long a two-hand gesture must be held before it fires.

    Deliberately much longer than ``arm_seconds``. Mode changes are disorienting
    and quitting is destructive, so both should require a pose you clearly meant
    to hold rather than one that flashed past while you were moving your hands."""

    two_hand_release_seconds: float = 0.30
    """Release tolerance for two-hand gestures, also longer than the single-hand
    value, so a brief loss of one hand does not re-arm them mid-hold."""


@dataclass
class SwitchingConfig:
    """Mode behaviour."""

    start: str = ""
    """Mode to start in, by name. Empty means the first configured mode."""

    wrap: bool = True
    """Whether "next mode" past the last mode returns to the first. When false it
    stops at the ends, which some people prefer so the list has a definite end."""

    announce_seconds: float = 1.8
    """How long the "mode changed" banner stays on screen."""

    settle_seconds: float = 0.5
    """Single-hand gestures are ignored for this long after a mode change.

    Without it, the pose you used to switch modes is immediately re-read against
    the *new* mode's table: both palms open changes the mode, and the primary
    hand's open palm would then fire whatever the new mode binds to an open palm,
    all in the same moment. The settle window lets you break the pose first."""

    allow_while_paused: bool = True
    """Whether mode switching and quitting work while paused.

    On by default: a pause should stop the controller acting on your machine, not
    lock you out of the controller. Set false if you want a pause that ignores
    every gesture, including the one that resumes."""

    primary_hand: str = "right"
    """Which hand drives single-hand gestures when both are visible:
    ``"right"``, ``"left"`` or ``"any"`` (whichever is seen first)."""

    single_hand_with_two_visible: str = "primary"
    """What single-hand gestures do while both hands are in frame.

    * ``"primary"``  -- only the :attr:`primary_hand` counts (the default);
    * ``"suppress"`` -- nothing fires from one hand while two are visible, the
      most conservative choice;
    * ``"both"``     -- either hand may fire, first match wins.

    The default avoids the surprising case where moving both hands into a
    two-hand shape passes through a single-hand shape and fires an action on the
    way, while still letting one hand do useful work with the other resting in
    frame."""

    def validate(self) -> None:
        if self.primary_hand.strip().lower() not in ("left", "right", "any"):
            raise ConfigError(
                f"switching.primary_hand must be 'left', 'right' or 'any', "
                f"got {self.primary_hand!r}"
            )
        allowed = ("primary", "suppress", "both")
        if self.single_hand_with_two_visible.strip().lower() not in allowed:
            raise ConfigError(
                f"switching.single_hand_with_two_visible must be one of "
                f"{', '.join(allowed)}, got {self.single_hand_with_two_visible!r}"
            )

    def start_index(self, names: Sequence[str]) -> int:
        """Resolve :attr:`start` to an index, raising a readable error."""
        if not self.start:
            return 0
        wanted = self.start.strip().lower()
        for index, name in enumerate(names):
            if name.lower() == wanted:
                return index
        raise ConfigError(
            f"switching.start refers to unknown mode {self.start!r}; "
            f"available: {', '.join(names)}"
        )


@dataclass
class ThresholdConfig:
    """Pose-decision thresholds, in degrees and palm lengths.

    Defaults are calibrated against the synthetic hand model; the HUD shows the
    live values so they can be retuned for a particular hand or camera. See
    :class:`gesture_control.pose.FingerThresholds` for the measured margins.
    """

    finger_curl_up: float = 150.0
    finger_curl_down: float = 120.0
    finger_reach_up: float = 1.15
    finger_reach_down: float = 0.95
    thumb_curl_up: float = 150.0
    thumb_curl_down: float = 130.0
    thumb_abduction_up: float = 0.35
    thumb_abduction_down: float = 0.28

    def build(self) -> Thresholds:
        return Thresholds(
            finger=FingerThresholds(
                curl_up=self.finger_curl_up,
                curl_down=self.finger_curl_down,
                reach_up=self.finger_reach_up,
                reach_down=self.finger_reach_down,
            ),
            thumb=ThumbThresholds(
                curl_up=self.thumb_curl_up,
                curl_down=self.thumb_curl_down,
                abduction_up=self.thumb_abduction_up,
                abduction_down=self.thumb_abduction_down,
            ),
        )

    def validate(self) -> None:
        finger = self.build().finger
        thumb = self.build().thumb
        if finger.curl_down >= finger.curl_up:
            raise ConfigError(
                f"finger_curl_down ({finger.curl_down}) must be below "
                f"finger_curl_up ({finger.curl_up})"
            )
        if finger.reach_down >= finger.reach_up:
            raise ConfigError(
                f"finger_reach_down ({finger.reach_down}) must be below "
                f"finger_reach_up ({finger.reach_up})"
            )
        if thumb.curl_down >= thumb.curl_up:
            raise ConfigError(
                f"thumb_curl_down ({thumb.curl_down}) must be below "
                f"thumb_curl_up ({thumb.curl_up})"
            )
        if thumb.abduction_down >= thumb.abduction_up:
            raise ConfigError(
                f"thumb_abduction_down ({thumb.abduction_down}) must be below "
                f"thumb_abduction_up ({thumb.abduction_up})"
            )


@dataclass
class ActionsConfig:
    """What actions do when they fire."""

    volume_step: float = 0.05
    """Volume change per step, 0-1. Lower than the original's 0.1 because
    hold-to-repeat now provides the coarse movement."""

    brightness_step: int = 8
    brightness_display: int | None = None
    """Monitor index for brightness changes; ``None`` means all of them."""

    dry_run: bool = False
    """Log actions without performing them. Useful for testing mappings."""

    require_focus_title: str = ""
    """If set, keyboard/media actions only fire while the foreground window
    title contains this text. This is the guard against a gesture typing into
    whatever window happens to be focused."""

    allow_keyboard: bool = True
    """Set false to disable keyboard and media synthesis entirely while leaving
    volume and brightness gestures working."""


@dataclass
class OverlayConfig:
    """On-screen display settings."""

    show_window: bool = True
    show_skeleton: bool = True
    show_finger_states: bool = True
    show_guide: bool = True
    show_system_levels: bool = True
    show_timings: bool = True
    scale: float = 1.0
    window_title: str = "Hand Gesture Control"


@dataclass
class AppConfig:
    """The complete application configuration."""

    preset: str = "pro"
    log_level: str = "INFO"
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    switching: SwitchingConfig = field(default_factory=SwitchingConfig)
    modes: tuple[Mode, ...] = ()
    two_hand: tuple[TwoHandGesture, ...] = ()

    def __post_init__(self) -> None:
        if not self.modes:
            preset = PRESETS.get(self.preset)
            if preset is None:
                raise ConfigError(
                    f"unknown preset {self.preset!r}; available: {', '.join(sorted(PRESETS))}"
                )
            self.modes = build_modes(preset.modes)
            if not self.two_hand:
                self.two_hand = build_two_hand_gestures(preset.two_hand)

    # -- derived views ----------------------------------------------------- #

    def build_thresholds(self) -> Thresholds:
        return self.thresholds.build()

    def build_mode_set(self) -> ModeSet:
        """Validate the mode table, raising a readable error if it is bad."""
        try:
            mode_set = ModeSet(modes=tuple(self.modes), wrap=self.switching.wrap)
        except ModeError as exc:
            raise ConfigError(f"invalid modes: {exc}") from exc
        return mode_set

    def build_two_hand_set(self) -> TwoHandGestureSet:
        try:
            return TwoHandGestureSet(gestures=tuple(self.two_hand))
        except ModeError as exc:
            raise ConfigError(f"invalid two-hand gestures: {exc}") from exc

    def build_gesture_set(self) -> GestureSet | None:
        """Gesture set of the *first* mode.

        Retained because a lot of tooling (and the self-test) wants "the gestures"
        for a single-mode configuration. Prefer :meth:`build_mode_set`.
        """
        return self.build_mode_set()[0].build_gesture_set()

    @property
    def mode_names(self) -> tuple[str, ...]:
        return tuple(mode.name for mode in self.modes)

    def validate(self) -> None:
        """Check the whole config, raising :class:`ConfigError` on any problem."""
        self.thresholds.validate()
        if self.detector.max_hands < 1:
            raise ConfigError("detector.max_hands must be at least 1")
        if self.detector.max_hands < 2 and self.two_hand:
            raise ConfigError(
                f"{len(self.two_hand)} two-hand gesture(s) are configured but "
                f"detector.max_hands is {self.detector.max_hands}; set it to 2 "
                f"(or remove the two-hand gestures), because a second hand can "
                f"never be detected otherwise"
            )
        if self.detector.stride < 1:
            raise ConfigError("detector.stride must be at least 1")
        if not 0.0 < self.detector.min_detection_confidence <= 1.0:
            raise ConfigError("detector.min_detection_confidence must be in (0, 1]")
        if not 0.0 < self.detector.min_tracking_confidence <= 1.0:
            raise ConfigError("detector.min_tracking_confidence must be in (0, 1]")
        if self.detector.model_complexity not in (0, 1):
            raise ConfigError("detector.model_complexity must be 0 or 1")
        if not 0.0 < self.actions.volume_step <= 1.0:
            raise ConfigError("actions.volume_step must be in (0, 1]")
        if not 1 <= self.actions.brightness_step <= 100:
            raise ConfigError("actions.brightness_step must be in [1, 100]")
        for name in ("arm_seconds", "release_seconds", "two_hand_arm_seconds",
                     "two_hand_release_seconds"):
            if getattr(self.tracking, name) < 0.0:
                raise ConfigError(f"tracking.{name} must not be negative")
        if self.tracking.repeat_interval <= 0.0:
            raise ConfigError("tracking.repeat_interval must be positive")
        if self.overlay.scale <= 0.0:
            raise ConfigError("overlay.scale must be positive")
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ConfigError("camera.width and camera.height must be positive")

        mode_set = self.build_mode_set()
        self.build_two_hand_set()
        self.switching.validate()
        # Resolves switching.start and raises a readable error if it is unknown.
        self.switching.start_index(mode_set.names)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #


def _gesture(
    name: str,
    action: str,
    requirements: str,
    *,
    label: str = "",
    description: str = "",
    repeatable: bool | None = None,
    cooldown: float | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "action": action,
        "requirements": requirements,
        "label": label or name.upper(),
        "description": description,
    }
    if repeatable is not None:
        payload["repeatable"] = repeatable
    if cooldown is not None:
        payload["cooldown"] = cooldown
    return payload



# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Preset:
    """A named starting configuration: its modes and its two-hand gestures."""

    name: str
    description: str
    modes: tuple[Mapping[str, Any], ...]
    two_hand: tuple[Mapping[str, Any], ...] = ()
    start_mode: str = ""


def _single_mode_preset(
    name: str,
    label: str,
    description: str,
    gestures: tuple[Mapping[str, Any], ...],
    *,
    accent: str = "accent",
) -> Preset:
    """A preset with exactly one mode, used by the legacy-style presets.

    Those still get the two-hand pause and quit gestures: the escape hatches
    should not depend on which preset is loaded.
    """
    return Preset(
        name=name,
        description=description,
        modes=(
            {
                "name": name,
                "label": label,
                "accent": accent,
                "description": description,
                "gestures": list(gestures),
            },
        ),
        two_hand=two_hand_basic(),
        start_mode=name,
    )


#: ``classic`` is a faithful reproduction of the original script's mapping:
#: fist -> space, index -> brightness up, index+middle -> left arrow,
#: pinky -> volume down, ring+pinky -> volume up. The only intentional change is
#: that the thumb is ignored for the four-finger gestures (the original required
#: it to be down, which made them fail whenever the thumb drifted) and that the
#: two-hand pause/quit gestures are available.
CLASSIC_GESTURES: tuple[Mapping[str, Any], ...] = (
    _gesture(
        "space", "key_space", "thumb:down,index:down,middle:down,ring:down,pinky:down",
        label="FIST",
        description="Closed fist presses the space bar (play/pause in most players).",
        repeatable=False,
    ),
    _gesture(
        "left", "key_left", "index:up,middle:up,ring:down,pinky:down",
        label="INDEX+MIDDLE", description="Two fingers press the left arrow key.",
        repeatable=True,
    ),
    _gesture(
        "volume_up", "volume_up", "ring:up,pinky:up,index:down,middle:down",
        label="RING+PINKY", description="Ring and little finger raise the volume.",
        repeatable=True,
    ),
    _gesture(
        "brightness_up", "brightness_up", "index:up,middle:down,ring:down,pinky:down",
        label="INDEX", description="Index finger alone raises screen brightness.",
        repeatable=True,
    ),
    _gesture(
        "volume_down", "volume_down", "pinky:up,index:down,middle:down,ring:down",
        label="PINKY", description="Little finger alone lowers the volume.",
        repeatable=True,
    ),
)

#: ``media`` trades the arrow keys and brightness for media-transport control.
MEDIA_GESTURES: tuple[Mapping[str, Any], ...] = (
    _gesture(
        "play_pause", "key_play_pause", "thumb:up,index:down,middle:down,ring:down,pinky:down",
        label="THUMB UP", description="Thumbs up toggles play/pause.",
        repeatable=False, cooldown=0.6,
    ),
    _gesture(
        "volume_up", "volume_up", "index:up,middle:down,ring:down,pinky:down",
        label="INDEX", description="Index finger alone raises the volume.",
        repeatable=True,
    ),
    _gesture(
        "volume_down", "volume_down", "index:up,middle:up,ring:down,pinky:down",
        label="INDEX+MIDDLE", description="Index and middle finger lower the volume.",
        repeatable=True,
    ),
    _gesture(
        "next_track", "next_track", "pinky:up,index:down,middle:down,ring:down",
        label="PINKY", description="Little finger alone skips to the next track.",
        repeatable=False, cooldown=0.8,
    ),
    _gesture(
        "previous_track", "previous_track", "ring:up,index:down,middle:down,pinky:down",
        label="RING", description="Ring finger alone goes to the previous track.",
        repeatable=False, cooldown=0.8,
    ),
    _gesture(
        "mute", "mute_toggle", "thumb:down,index:down,middle:down,ring:down,pinky:down",
        label="FIST", description="Closed fist toggles mute.",
        repeatable=False, cooldown=0.6,
    ),
    _gesture(
        "open_palm", "key_play_pause", "thumb:up,index:up,middle:up,ring:up,pinky:up",
        label="OPEN PALM", description="Open palm toggles play/pause.",
        repeatable=False, cooldown=0.8,
    ),
)

#: ``minimal`` keeps the original's hand shapes but maps them to the actions
#: that make most sense for those shapes.
MINIMAL_GESTURES: tuple[Mapping[str, Any], ...] = (
    _gesture(
        "mute", "mute_toggle", "thumb:down,index:down,middle:down,ring:down,pinky:down",
        label="FIST", description="Closed fist toggles mute.",
        repeatable=False, cooldown=0.6,
    ),
    _gesture(
        "volume_up", "volume_up", "index:up,middle:down,ring:down,pinky:down",
        label="INDEX", description="Index finger alone raises the volume.",
        repeatable=True,
    ),
    _gesture(
        "volume_down", "volume_down", "index:up,middle:up,ring:down,pinky:down",
        label="INDEX+MIDDLE", description="Index and middle finger lower the volume.",
        repeatable=True,
    ),
    _gesture(
        "brightness_up", "brightness_up", "ring:up,pinky:up,index:down,middle:down",
        label="RING+PINKY", description="Ring and little finger raise brightness.",
        repeatable=True,
    ),
    _gesture(
        "brightness_down", "brightness_down", "pinky:up,index:down,middle:down,ring:down",
        label="PINKY", description="Little finger alone lowers brightness.",
        repeatable=True,
    ),
)

#: ``present`` is a single-mode presentation remote, for people who want the
#: slide controls without the mode layer.
PRESENT_GESTURES: tuple[Mapping[str, Any], ...] = (
    _gesture("next_slide", "key_right", shape_table("index"), label="INDEX", repeatable=True),
    _gesture("prev_slide", "key_left", shape_table("peace"), label="PEACE", repeatable=True),
    _gesture("start", "key_start_slideshow", shape_table("open"), label="OPEN PALM", repeatable=False),
    _gesture("blank", "key_blank_screen", shape_table("fist"), label="FIST", repeatable=False),
    _gesture("end", "key_escape", shape_table("pinky"), label="PINKY", repeatable=False),
)

PRESETS: Mapping[str, Preset] = {
    "pro": Preset(
        name="pro",
        description=(
            "Five modes (media, system, browse, present, window) selected with "
            "two hands. Same hand shapes, different actions per mode."
        ),
        modes=MODE_PRESETS,
        two_hand=two_hand_pro(),
        start_mode="media",
    ),
    "classic": _single_mode_preset(
        "classic",
        "CLASSIC",
        "The original script's mapping: space, left arrow, volume, brightness.",
        CLASSIC_GESTURES,
    ),
    "media": _single_mode_preset(
        "media",
        "MEDIA",
        "Media transport: play/pause, tracks, volume, mute.",
        MEDIA_GESTURES,
    ),
    "minimal": _single_mode_preset(
        "minimal",
        "MINIMAL",
        "Volume and brightness only, with mute on a fist.",
        MINIMAL_GESTURES,
    ),
    "present": _single_mode_preset(
        "present",
        "PRESENT",
        "A single-mode presentation remote: next, back, blank, start, end.",
        PRESENT_GESTURES,
        accent="index",
    ),
}


def available_presets() -> dict[str, str]:
    """Preset name -> one-line description, for ``--list-presets``."""
    return {name: preset.description for name, preset in PRESETS.items()}


def build_gestures(
    entries: Sequence[Mapping[str, Any] | GestureSpec],
) -> tuple[GestureSpec, ...]:
    """Turn preset/config gesture tables into :class:`GestureSpec` objects."""
    specs: list[GestureSpec] = []
    for entry in entries:
        if isinstance(entry, GestureSpec):
            specs.append(entry)
            continue
        try:
            name = str(entry["name"])
            action = str(entry["action"])
        except KeyError as exc:
            raise ConfigError(f"gesture entry is missing the {exc.args[0]!r} key") from exc
        raw_requirements = entry.get("requirements", entry.get("fingers", ""))
        try:
            requirements = parse_requirements(raw_requirements)
        except GestureError as exc:
            raise ConfigError(f"gesture {name!r}: {exc}") from exc
        specs.append(
            GestureSpec(
                name=name,
                action=action,
                requirements=requirements,
                label=str(entry.get("label", name.upper())),
                description=str(entry.get("description", "")),
                repeatable=entry.get("repeatable"),
                cooldown=entry.get("cooldown"),
            )
        )
    return tuple(specs)


def build_modes(
    entries: Sequence[Mode | Mapping[str, Any]],
) -> tuple[Mode, ...]:
    """Turn preset/config mode tables into :class:`Mode` objects."""
    modes: list[Mode] = []
    for entry in entries:
        if isinstance(entry, Mode):
            modes.append(entry)
            continue
        if "name" not in entry:
            raise ConfigError("a [[modes]] entry needs a name")
        modes.append(
            Mode(
                name=str(entry["name"]),
                label=str(entry.get("label") or ""),
                description=str(entry.get("description") or ""),
                gestures=build_gestures(_as_list(entry.get("gestures"), "gestures")),
                accent=str(entry.get("accent") or "accent"),
            )
        )
    return tuple(modes)


def build_two_hand_gestures(
    entries: Sequence[TwoHandGesture | Mapping[str, Any]],
) -> tuple[TwoHandGesture, ...]:
    """Turn preset/config two-hand tables into :class:`TwoHandGesture` objects."""
    built: list[TwoHandGesture] = []
    for entry in entries:
        if isinstance(entry, TwoHandGesture):
            built.append(entry)
            continue
        try:
            name = str(entry["name"])
            action = str(entry["action"])
        except KeyError as exc:
            raise ConfigError(
                f"a [[two_hand]] entry is missing the {exc.args[0]!r} key"
            ) from exc

        def _table(key: str) -> dict[Any, Any] | None:
            raw = entry.get(key)
            if raw is None:
                return None
            try:
                return parse_requirements(raw)
            except GestureError as exc:
                raise ConfigError(f"two-hand gesture {name!r}: {exc}") from exc

        try:
            built.append(
                TwoHandGesture(
                    name=name,
                    action=action,
                    shape=_table("shape"),
                    left=_table("left"),
                    right=_table("right"),
                    label=str(entry.get("label") or ""),
                    description=str(entry.get("description") or ""),
                    repeatable=entry.get("repeatable"),
                    cooldown=entry.get("cooldown"),
                )
            )
        except ModeError as exc:
            raise ConfigError(str(exc)) from exc
    return tuple(built)


def _as_list(value: Any, what: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{what} must be a list of tables")
    return value


def default_config(preset: str = "pro") -> AppConfig:
    """Build a default configuration, optionally based on a preset."""
    chosen = PRESETS.get(preset)
    if chosen is None:
        raise ConfigError(
            f"unknown preset {preset!r}; available: {', '.join(sorted(PRESETS))}"
        )
    return AppConfig(
        preset=preset,
        modes=build_modes(chosen.modes),
        two_hand=build_two_hand_gestures(chosen.two_hand),
        switching=SwitchingConfig(start=chosen.start_mode),
    )


# --------------------------------------------------------------------------- #
# Loading and merging
# --------------------------------------------------------------------------- #


def _unwrap_optional(annotation: Any) -> Any:
    """Reduce ``X | None`` to ``X``; pass other annotations through."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Coerce a TOML value to the declared field type, with a readable error."""
    target = _unwrap_optional(annotation)
    if value is None:
        return None
    if target is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _apply_section(target: Any, data: Mapping[str, Any], path: str) -> None:
    """Recursively merge ``data`` into a dataclass instance, rejecting typos.

    Field annotations are resolved via :func:`typing.get_type_hints` because
    ``from __future__ import annotations`` turns every ``Field.type`` into a
    string. Resolution is done once per section, keyed by field *name*.
    """
    import typing

    fields = {f.name: f for f in dataclasses.fields(target)}
    try:
        hints = typing.get_type_hints(type(target))
    except Exception:  # pragma: no cover - defensive; annotations are static
        hints = {}

    for key, value in data.items():
        if key not in fields:
            valid = ", ".join(sorted(fields))
            raise ConfigError(f"{path}: unknown option {key!r}. Valid options: {valid}")
        current = getattr(target, key)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            _apply_section(current, value, f"{path}.{key}" if path else key)
            continue
        # Structured tables need their own builder rather than a scalar coercion.
        if key == "modes":
            if not isinstance(value, list):
                raise ConfigError(f"{path}.modes must be a list of [[modes]] tables")
            setattr(target, key, build_modes(value))
            continue
        if key == "two_hand":
            if not isinstance(value, list):
                raise ConfigError(f"{path}.two_hand must be a list of [[two_hand]] tables")
            setattr(target, key, build_two_hand_gestures(value))
            continue
        if key == "gestures":
            # Only reachable for a legacy top-level list; handled in load_config,
            # but kept here so a nested mode table behaves sensibly too.
            if not isinstance(value, list):
                raise ConfigError(f"{path}.gestures must be a list of tables")
            setattr(target, key, build_gestures(value))
            continue
        if value is None:
            setattr(target, key, None)
            continue
        annotation = hints.get(key, fields[key].type)
        setattr(target, key, _coerce(value, annotation, f"{path}.{key}" if path else key))


def _read_config_text(path: Path) -> str:
    """Read a config file, tolerating the encodings Windows shells produce.

    ``python -m gesture_control --print-config > config.toml`` in PowerShell
    writes UTF-16LE with a byte-order mark, not UTF-8, so a strictly UTF-8 reader
    fails on a file that looks perfectly fine in an editor. Trying the plausible
    encodings in order turns that from a confusing crash into something that
    works, and the error message names the encodings tried if none fit.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ConfigError(
        f"{path} could not be decoded as text (tried utf-8, utf-16 and cp1252). "
        f"It may be a binary file, or a shortcut rather than a TOML file."
    )


def load_config(path: str | Path | None = None, *, preset: str | None = None) -> AppConfig:
    """Load configuration from a TOML file, layered over defaults and a preset.

    Args:
        path: TOML file, or ``None`` for defaults only.
        preset: preset to start from before applying the file. When given, it
            overrides any ``preset`` key in the file -- an explicit command-line
            choice should not be silently reversed by the file it loads.

    Raises:
        ConfigError: on malformed TOML, unknown keys, wrong types, or an invalid
            resulting configuration.
    """
    data: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
        try:
            data = tomllib.loads(_read_config_text(config_path))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{config_path} is not valid TOML: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"could not read {config_path}: {exc}") from exc

    file_preset = data.pop("preset", None)
    if file_preset is not None and not isinstance(file_preset, str):
        raise ConfigError(f"preset must be a string, got {file_preset!r}")
    if file_preset is not None and file_preset not in PRESETS:
        raise ConfigError(
            f"unknown preset {file_preset!r}; available: {', '.join(sorted(PRESETS))}"
        )

    chosen = preset or file_preset or "pro"
    if chosen not in PRESETS:
        raise ConfigError(f"unknown preset {chosen!r}; available: {', '.join(sorted(PRESETS))}")

    # Legacy compatibility: a config written before modes existed has a top-level
    # [[gestures]] list. Wrap it in a single mode rather than refusing to load it,
    # so an existing file keeps working.
    legacy_gestures = data.pop("gestures", None)
    if legacy_gestures is not None:
        if "modes" in data:
            raise ConfigError(
                "this file defines both [[gestures]] (the old single-mode format) "
                "and [[modes]]; keep only [[modes]]"
            )
        if not isinstance(legacy_gestures, list) or not legacy_gestures:
            raise ConfigError(
                "gestures must be a non-empty list of [[gestures]] tables; "
                "remove the key entirely to use the preset's gestures"
            )
        data["modes"] = [
            {
                "name": "default",
                "label": "DEFAULT",
                "description": "Legacy [[gestures]] list from a pre-modes config.",
                "gestures": legacy_gestures,
            }
        ]
        _LOG.warning(
            "this config uses the old top-level [[gestures]] format; it has been "
            "loaded as a single mode named 'default'. See config.example.toml for "
            "the [[modes]] format."
        )

    config = default_config(chosen)

    # If the file supplies its own modes, the preset's start mode may no longer
    # exist. Clear it unless the file also chose one, so a custom mode list does
    # not fail with a confusing "unknown mode 'media'" for a mode the user never
    # mentioned.
    if "modes" in data and not (
        isinstance(data.get("switching"), Mapping) and "start" in data["switching"]
    ):
        config.switching.start = ""

    _apply_section(config, data, "")
    config.validate()
    return config


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

_TOML_BARE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _toml_key(name: str) -> str:
    return name if all(ch in _TOML_BARE for ch in name) else _toml_quote(name)


def _toml_quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, 6))
    if isinstance(value, str):
        return _toml_quote(value)
    raise ConfigError(f"cannot serialise {value!r} to TOML")


#: Sections rendered as plain ``[name]`` tables of scalars.
_SCALAR_SECTIONS: tuple[str, ...] = (
    "camera",
    "detector",
    "smoothing",
    "tracking",
    "thresholds",
    "actions",
    "overlay",
    "switching",
)


def config_to_toml(config: AppConfig) -> str:
    """Render a config as TOML that loads back to an equal configuration."""
    lines: list[str] = [
        "# Hand Gesture Control configuration",
        "# Generated by: python -m gesture_control --print-config",
        "#",
        "# Precedence: these values override the selected preset; command-line",
        "# flags override this file. Unknown keys are an error, not a warning.",
        "#",
        "# Modes: each [mode] owns its own gesture table, so the same hand shape",
        "# can do different things in different modes. Switch modes with a",
        "# two-hand gesture (see [[two_hand]] at the bottom).",
        "",
        f"preset = {_toml_value(config.preset)}",
        f"log_level = {_toml_value(config.log_level)}",
        "",
    ]

    for section in _SCALAR_SECTIONS:
        instance = getattr(config, section)
        lines.append(f"[{section}]")
        lines.extend(_render_table(instance))
        lines.append("")

    lines.append("# requirements: 'up', 'down' or 'any' per finger; 'any' means the")
    lines.append("# gesture does not care about that finger. Gestures within a mode are")
    lines.append("# validated for ambiguity at startup.")
    for mode in config.modes:
        lines.append("")
        lines.append("[[modes]]")
        lines.append(f"name = {_toml_value(mode.name)}")
        lines.append(f"label = {_toml_value(mode.label)}")
        lines.append(f"accent = {_toml_value(mode.accent)}")
        if mode.description:
            lines.append(f"description = {_toml_value(mode.description)}")
        for spec in mode.gestures:
            lines.append("")
            lines.append("[[modes.gestures]]")
            lines.extend(_render_gesture(spec))

    lines.append("")
    lines.append("# Two-hand gestures. Both hands are required, which is what makes them")
    lines.append("# safe to bind to mode changes and to quitting: they cannot be triggered")
    lines.append("# while gesturing with one hand. Use shape = for a symmetric gesture")
    lines.append("# (the same hand shape on both hands) or left =/right = for an ordered one.")
    for gesture in config.two_hand:
        lines.append("")
        lines.append("[[two_hand]]")
        lines.append(f"name = {_toml_value(gesture.name)}")
        lines.append(f"action = {_toml_value(gesture.action)}")
        lines.append(f"label = {_toml_value(gesture.label)}")
        if gesture.symmetric:
            lines.append(f"shape = {_toml_value(_requirements_string(gesture.shape or {}))}")
        else:
            if gesture.left:
                lines.append(f"left = {_toml_value(_requirements_string(gesture.left))}")
            if gesture.right:
                lines.append(f"right = {_toml_value(_requirements_string(gesture.right))}")
        if gesture.description:
            lines.append(f"description = {_toml_value(gesture.description)}")
        if gesture.repeatable is not None:
            lines.append(f"repeatable = {_toml_value(bool(gesture.repeatable))}")
        if gesture.cooldown is not None:
            lines.append(f"cooldown = {_toml_value(float(gesture.cooldown))}")
    lines.append("")
    return "\n".join(lines)


def _requirements_string(requirements: Mapping[Any, Any]) -> str:
    return ",".join(
        f"{finger.value}:{requirement.value}" for finger, requirement in requirements.items()
    )


def _render_gesture(spec: GestureSpec) -> list[str]:
    lines = [
        f"name = {_toml_value(spec.name)}",
        f"action = {_toml_value(spec.action)}",
        f"label = {_toml_value(spec.label)}",
        f"requirements = {_toml_value(_requirements_string(spec.requirements))}",
    ]
    if spec.description:
        lines.append(f"description = {_toml_value(spec.description)}")
    if spec.repeatable is not None:
        lines.append(f"repeatable = {_toml_value(bool(spec.repeatable))}")
    if spec.cooldown is not None:
        lines.append(f"cooldown = {_toml_value(float(spec.cooldown))}")
    return lines


def _render_table(instance: Any) -> list[str]:
    lines: list[str] = []
    for field_info in dataclasses.fields(instance):
        value = getattr(instance, field_info.name)
        if value is None:
            lines.append(f"# {field_info.name} = <unset>")
            continue
        lines.append(f"{_toml_key(field_info.name)} = {_toml_value(value)}")
    return lines


def iter_config_paths(config: AppConfig) -> Iterable[str]:
    """Dotted paths of every settable option, for documentation and tests."""
    for section in _SCALAR_SECTIONS:
        for field_info in dataclasses.fields(getattr(config, section)):
            yield f"{section}.{field_info.name}"
