"""Gesture vocabulary: named hand shapes and the per-mode gesture tables.

Kept separate from :mod:`gesture_control.config` because it is *content* rather
than plumbing, and it is long enough to drown the dataclasses.

Two conventions matter here:

**Shapes are fully specified.** Every shape pins all five fingers to up or down.
That is a deliberate trade: it is stricter than the forgiving ``thumb:any`` used
by the ``classic`` preset, but it makes ambiguity *structurally impossible* --
two different complete shapes always disagree about at least one finger, so no
two gestures in a mode can ever both match the same hand. With a vocabulary this
size, that guarantee is worth more than the extra tolerance.

**Two-hand gestures are symmetric where possible.** A symmetric gesture ignores
which hand is which, so a mislabelled handedness cannot silently disable it.
Ordered gestures (``left``/``right``) exist for combinations such as "left open,
right fist" and still tolerate a swapped label by trying both assignments.
"""

from __future__ import annotations

from typing import Mapping

from .gestures import Requirement

__all__ = [
    "SHAPES",
    "TWO_HAND_OPEN",
    "TWO_HAND_FIST",
    "shape_table",
    "MODE_PRESETS",
]

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: Named, fully specified hand shapes. The key is used in the documentation and
#: in ``--list-gestures``; the value is the requirement string the config format
#: already understands.
SHAPES: Mapping[str, str] = {
    "fist": "thumb:down,index:down,middle:down,ring:down,pinky:down",
    "open": "thumb:up,index:up,middle:up,ring:up,pinky:up",
    "index": "thumb:down,index:up,middle:down,ring:down,pinky:down",
    "peace": "thumb:down,index:up,middle:up,ring:down,pinky:down",
    "three": "thumb:up,index:up,middle:up,ring:down,pinky:down",
    "three_inner": "thumb:down,index:up,middle:up,ring:up,pinky:down",
    "four": "thumb:down,index:up,middle:up,ring:up,pinky:up",
    "claw": "thumb:up,index:up,middle:up,ring:up,pinky:down",
    "pinky": "thumb:down,index:down,middle:down,ring:down,pinky:up",
    "ring_pinky": "thumb:down,index:down,middle:down,ring:up,pinky:up",
    "mid_ring_pinky": "thumb:down,index:down,middle:up,ring:up,pinky:up",
    "thumb_up": "thumb:up,index:down,middle:down,ring:down,pinky:down",
    "thumb_middle": "thumb:up,index:down,middle:up,ring:down,pinky:down",
    "shaka": "thumb:up,index:down,middle:down,ring:down,pinky:up",
    "horns": "thumb:down,index:up,middle:down,ring:down,pinky:up",
    "gun": "thumb:up,index:up,middle:down,ring:down,pinky:down",
}

#: ASCII art per shape, for the printed reference.
SHAPE_ART: Mapping[str, str] = {
    "fist": "closed hand",
    "open": "all five out",
    "index": "one finger",
    "peace": "V sign",
    "three": "thumb + two",
    "three_inner": "three, no thumb",
    "four": "four, no thumb",
    "claw": "four + thumb",
    "pinky": "little finger",
    "ring_pinky": "two, outer",
    "mid_ring_pinky": "three, outer",
    "thumb_up": "thumbs up",
    "thumb_middle": "thumb + middle",
    "shaka": "thumb + little",
    "horns": "index + little",
    "gun": "thumb + index",
}


def shape_table(name: str) -> str:
    """Requirement string for a named shape, with a helpful error otherwise."""
    try:
        return SHAPES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown shape {name!r}; known shapes: {', '.join(sorted(SHAPES))}"
        ) from exc


# --------------------------------------------------------------------------- #
# Mode tables
# --------------------------------------------------------------------------- #


def _gesture(
    name: str,
    action: str,
    shape: str,
    *,
    label: str = "",
    description: str = "",
    repeatable: bool | None = None,
    cooldown: float | None = None,
) -> dict[str, object]:
    """One gesture entry, using a named shape."""
    entry: dict[str, object] = {
        "name": name,
        "action": action,
        "requirements": shape_table(shape),
        "label": label or name.upper(),
        "description": description or SHAPE_ART.get(shape, ""),
    }
    if repeatable is not None:
        entry["repeatable"] = repeatable
    if cooldown is not None:
        entry["cooldown"] = cooldown
    return entry


#: Every mode ends with the same pause gesture, deliberately. A pause must be
#: reversible in whichever mode the user happens to be in, so the escape hatch
#: cannot be mode-dependent.
_PAUSE = _gesture(
    "pause",
    "pause_toggle",
    "shaka",
    label="PAUSE",
    description="Thumb and little finger: pause or resume recognition.",
)


def _mode(
    name: str,
    label: str,
    accent: str,
    description: str,
    gestures: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "name": name,
        "label": label,
        "accent": accent,
        "description": description,
        "gestures": [*gestures, _PAUSE],
    }


MEDIA_MODE = _mode(
    "media",
    "MEDIA",
    "accent",
    "Playback: transport, tracks and volume.",
    (
        _gesture("play_pause", "key_play_pause", "open", label="OPEN PALM"),
        _gesture("mute", "mute_toggle", "fist", label="FIST"),
        _gesture("volume_up", "volume_up", "index", label="INDEX", repeatable=True),
        _gesture("volume_down", "volume_down", "peace", label="PEACE", repeatable=True),
        _gesture("next", "next_track", "pinky", label="PINKY", repeatable=False),
        _gesture("prev", "previous_track", "ring_pinky", label="RING+PINKY", repeatable=False),
        _gesture("stop", "stop_media", "thumb_up", label="THUMB UP", repeatable=False),
        _gesture("media_vol_up", "media_volume_up", "four", label="FOUR", repeatable=True),
        _gesture("media_vol_down", "media_volume_down", "claw", label="CLAW", repeatable=True),
    ),
)

SYSTEM_MODE = _mode(
    "system",
    "SYSTEM",
    "warn",
    "Machine controls: display, sound and windows.",
    (
        _gesture("brightness_up", "brightness_up", "index", label="INDEX", repeatable=True),
        _gesture("brightness_down", "brightness_down", "peace", label="PEACE", repeatable=True),
        _gesture("volume_down", "volume_down", "pinky", label="PINKY", repeatable=True),
        _gesture("volume_up", "volume_up", "ring_pinky", label="RING+PINKY", repeatable=True),
        _gesture("mute", "mute_toggle", "fist", label="FIST"),
        _gesture("show_desktop", "key_show_desktop", "open", label="OPEN PALM", repeatable=False),
        _gesture("task_view", "key_task_view", "thumb_up", label="THUMB UP", repeatable=False),
        _gesture("switch_window", "key_alt_tab", "gun", label="GUN", repeatable=False),
    ),
)

BROWSE_MODE = _mode(
    "browse",
    "BROWSE",
    "ok",
    "Reading and browsing: pages, tabs and history.",
    (
        _gesture("page_down", "key_page_down", "index", label="INDEX", repeatable=True),
        _gesture("page_up", "key_page_up", "peace", label="PEACE", repeatable=True),
        _gesture("next_tab", "next_tab", "pinky", label="PINKY", repeatable=False),
        _gesture("prev_tab", "prev_tab", "ring_pinky", label="RING+PINKY", repeatable=False),
        _gesture("stop", "key_escape", "thumb_up", label="THUMB UP", repeatable=False),
        _gesture("select", "key_enter", "fist", label="FIST", repeatable=False),
        _gesture("top", "key_home", "open", label="OPEN PALM", repeatable=False),
        _gesture("back", "key_back", "horns", label="HORNS", repeatable=True, cooldown=0.6),
        _gesture("forward", "key_forward", "gun", label="GUN", repeatable=True, cooldown=0.6),
    ),
)

PRESENT_MODE = _mode(
    "present",
    "PRESENT",
    "index",
    "Slides: advance, go back, blank the screen.",
    (
        _gesture("next_slide", "key_right", "index", label="INDEX", repeatable=True),
        _gesture("prev_slide", "key_left", "peace", label="PEACE", repeatable=True),
        _gesture("start", "key_start_slideshow", "open", label="OPEN PALM", repeatable=False),
        _gesture("blank", "key_blank_screen", "fist", label="FIST", repeatable=False),
        _gesture("end", "key_escape", "pinky", label="PINKY", repeatable=False),
        _gesture("first", "key_home", "thumb_up", label="THUMB UP", repeatable=False),
        _gesture("last", "key_end", "ring_pinky", label="RING+PINKY", repeatable=False),
        _gesture("fullscreen", "key_fullscreen", "four", label="FOUR", repeatable=False),
    ),
)

WINDOW_MODE = _mode(
    "window",
    "WINDOW",
    "thumb",
    "Windows, desktops and screenshots.",
    (
        _gesture("switch", "key_alt_tab", "thumb_up", label="THUMB UP", repeatable=False),
        _gesture("task_view", "key_task_view", "open", label="OPEN PALM", repeatable=False),
        _gesture("show_desktop", "key_show_desktop", "fist", label="FIST", repeatable=False),
        _gesture("confirm", "key_enter", "index", label="INDEX", repeatable=False),
        _gesture("dismiss", "key_escape", "peace", label="PEACE", repeatable=False),
        _gesture("fullscreen", "key_fullscreen", "pinky", label="PINKY", repeatable=False),
        _gesture("screenshot", "key_screenshot", "ring_pinky", label="RING+PINKY", repeatable=False),
        _gesture("tab", "key_tab", "four", label="FOUR", repeatable=True, cooldown=0.3),
    ),
)

#: Mode tables for the ``pro`` preset, in switching order.
MODE_PRESETS: tuple[Mapping[str, object], ...] = (
    MEDIA_MODE,
    SYSTEM_MODE,
    BROWSE_MODE,
    PRESENT_MODE,
    WINDOW_MODE,
)

#: Two-hand shapes, expressed as named shapes for both hands.
TWO_HAND_OPEN = shape_table("open")
TWO_HAND_FIST = shape_table("fist")


def two_hand_pro() -> tuple[dict[str, object], ...]:
    """Two-hand gestures for the multi-mode presets.

    Each is a deliberate two-handed act:

    * both open palms -- next mode;
    * both fists -- previous mode;
    * both peace signs -- jump back to the first mode;
    * both shaka -- pause or resume, from any mode;
    * both horns -- quit.

    Quit and pause are the ones worth having on both hands: one is destructive
    and the other would otherwise be a trap.
    """
    return (
        {
            "name": "mode_next",
            "action": "mode_next",
            "shape": TWO_HAND_OPEN,
            "label": "BOTH OPEN",
            "description": "Both palms open: next mode.",
            "repeatable": False,
        },
        {
            "name": "mode_prev",
            "action": "mode_prev",
            "shape": TWO_HAND_FIST,
            "label": "BOTH FISTS",
            "description": "Both hands closed: previous mode.",
            "repeatable": False,
        },
        {
            "name": "mode_home",
            "action": "mode_home",
            "shape": shape_table("peace"),
            "label": "BOTH PEACE",
            "description": "Both V signs: jump back to the first mode.",
            "repeatable": False,
        },
        {
            "name": "pause",
            "action": "pause_toggle",
            "shape": shape_table("shaka"),
            "label": "BOTH SHAKA",
            "description": "Both hands thumb+little: pause or resume.",
            "repeatable": False,
        },
        {
            "name": "quit",
            "action": "quit_app",
            "shape": shape_table("horns"),
            "label": "BOTH HORNS",
            "description": "Both hands index+little: quit the controller.",
            "repeatable": False,
        },
    )


def two_hand_basic() -> tuple[dict[str, object], ...]:
    """Two-hand gestures for the single-mode presets: pause and quit only.

    Mode switching would be meaningless with one mode, but a quit gesture and a
    reversible pause are wanted everywhere.
    """
    return (
        {
            "name": "pause",
            "action": "pause_toggle",
            "shape": shape_table("shaka"),
            "label": "BOTH SHAKA",
            "description": "Both hands thumb+little: pause or resume.",
            "repeatable": False,
        },
        {
            "name": "quit",
            "action": "quit_app",
            "shape": shape_table("horns"),
            "label": "BOTH HORNS",
            "description": "Both hands index+little: quit the controller.",
            "repeatable": False,
        },
    )


def requirement_of(shape_name: str) -> Mapping[str, str]:
    """Parse a named shape into a ``{finger: state}`` mapping (for tests)."""
    from .gestures import parse_requirements

    return {f.value: r.value for f, r in parse_requirements(shape_table(shape_name)).items()}


# Re-exported so callers do not need to reach into gestures for the enum.
REQUIREMENT = Requirement
