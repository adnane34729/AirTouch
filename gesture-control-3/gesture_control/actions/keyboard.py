"""Keyboard and media-key actions, with an optional focus guard.

The real risk in this project is not a wrong volume step -- it is a gesture that
synthesises a keystroke into whatever window happens to be focused. Alt-tab away
from the preview and a fist can press space in a document, a left arrow in a
spreadsheet, or worse.

Two mitigations:

* ``actions.allow_keyboard = false`` disables keyboard synthesis entirely while
  leaving volume and brightness working.
* ``actions.require_focus_title = "..."`` makes keyboard actions fire only when
  the foreground window's title contains that text. Guarded actions report
  *why* they were skipped, so a gesture that does nothing is explainable rather
  than mysterious.

The foreground window is read with ``GetForegroundWindow``/``GetWindowTextW``
through ctypes, so no extra dependency is needed.
"""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from ..config import ActionsConfig
from .base import Action, ActionError, ActionOutcome

__all__ = [
    "foreground_window_title",
    "KeyPressAction",
    "MediaKeyAction",
    "build_keyboard_actions",
    "KEY_ACTION_SPECS",
]

_LOG = logging.getLogger(__name__)

WINDOWS = sys.platform == "win32"


def foreground_window_title() -> str | None:
    """Title of the focused window, or ``None`` if it cannot be determined.

    Returning ``None`` rather than raising matters for the focus guard: if the
    title cannot be read, the guard must decide to what to do, and it chooses to
    block (see :class:`_GuardedAction`).
    """
    if not WINDOWS:
        return None
    try:
        import ctypes

        user32 = ctypes.WinDLL("user32.dll")
        handle = user32.GetForegroundWindow()
        if not handle:
            return None
        length = user32.GetWindowTextLengthW(handle)
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        return buffer.value
    except Exception as exc:  # pragma: no cover - platform dependent
        _LOG.debug("could not read the foreground window: %s", exc)
        return None


class _KeyboardAction(Action):
    """Base class holding the focus guard and the pynput controller."""

    touches_system = True
    repeatable = False
    cooldown = 0.3

    def __init__(
        self,
        *,
        enabled: bool = True,
        require_focus_title: str = "",
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.require_focus_title = require_focus_title
        self._controller = None
        self.presses = 0

    # -- infrastructure ---------------------------------------------------- #

    def _controller_or_raise(self):
        if self._controller is None:
            try:
                from pynput.keyboard import Controller
            except Exception as exc:
                raise ActionError(
                    f"pynput is not available, so keys cannot be synthesised: {exc}"
                ) from exc
            self._controller = Controller()
        return self._controller

    def _focus_allows(self) -> tuple[bool, str]:
        """Apply the focus guard. Returns ``(allowed, explanation)``."""
        if not self.enabled:
            return False, "keyboard actions are disabled (actions.allow_keyboard = false)"
        if not self.require_focus_title:
            return True, ""
        title = foreground_window_title()
        if title is None:
            return False, "could not read the foreground window title"
        if self.require_focus_title.lower() not in title.lower():
            return False, f"focused window {title[:48]!r} is not the target window"
        return True, ""

    def _blocked(self) -> ActionOutcome | None:
        """A skipped-by-guard outcome, or ``None`` when the action may proceed.

        Blocking is an expected condition, not an error, so it is reported as a
        failed outcome rather than raised. The HUD then shows *why* nothing
        happened, which is the difference between a guard and a mystery.
        """
        allowed, reason = self._focus_allows()
        return None if allowed else ActionOutcome(ok=False, message="skipped", detail=reason)

    def describe(self) -> str:
        guard = f", focus={self.require_focus_title!r}" if self.require_focus_title else ""
        state = "enabled" if self.enabled else "disabled"
        return f"{self.id}: {self.label} [{state}{guard}]"


class KeyPressAction(_KeyboardAction):
    """Press and release one key (or a short sequence of keys)."""

    def __init__(
        self,
        action_id: str,
        key_names: Sequence[str],
        *,
        label: str = "",
        repeatable: bool = False,
        cooldown: float = 0.3,
        enabled: bool = True,
        require_focus_title: str = "",
    ) -> None:
        super().__init__(enabled=enabled, require_focus_title=require_focus_title)
        if not key_names:
            raise ActionError(f"{action_id} needs at least one key")
        self.id = action_id
        self.key_names = tuple(key_names)
        self.label = label or action_id.replace("key_", "").replace("_", " ").title()
        self.repeatable = repeatable
        self.cooldown = cooldown

    def _resolve(self, name: str):
        from pynput.keyboard import Key

        attribute = name.strip().lower()
        special = getattr(Key, attribute, None)
        if special is not None:
            return special
        if len(name) == 1:
            return name
        raise ActionError(
            f"unknown key {name!r} for action {self.id!r}; "
            f"use a single character or a pynput Key name such as 'left', 'space', 'f5'"
        )

    def invoke(self) -> ActionOutcome:
        blocked = self._blocked()
        if blocked is not None:
            return blocked
        controller = self._controller_or_raise()
        keys = [self._resolve(name) for name in self.key_names]
        for key in keys:
            controller.press(key)
        for key in reversed(keys):
            controller.release(key)
        self.presses += 1
        return ActionOutcome(
            ok=True,
            message=f"pressed {'+'.join(self.key_names)}",
            detail="pynput keyboard controller",
        )


class MediaKeyAction(_KeyboardAction):
    """Press a media-transport key (play/pause, next, previous, stop)."""

    def __init__(
        self,
        action_id: str,
        media_key: str,
        *,
        label: str = "",
        repeatable: bool = False,
        cooldown: float = 0.6,
        enabled: bool = True,
        require_focus_title: str = "",
    ) -> None:
        super().__init__(enabled=enabled, require_focus_title=require_focus_title)
        self.id = action_id
        self.media_key = media_key
        self.label = label or media_key.replace("media_", "").replace("_", " ").title()
        self.repeatable = repeatable
        self.cooldown = cooldown

    def invoke(self) -> ActionOutcome:
        # Media keys are system-wide by design, so the focus guard does not
        # apply to them; only the global keyboard switch does.
        blocked = self._blocked()
        if blocked is not None:
            return blocked
        from pynput.keyboard import Key

        key = getattr(Key, self.media_key, None)
        if key is None:
            raise ActionError(f"this pynput build has no Key.{self.media_key}")
        controller = self._controller_or_raise()
        controller.press(key)
        controller.release(key)
        self.presses += 1
        return ActionOutcome(ok=True, message=f"media key {self.media_key}",
                             detail="pynput keyboard controller")


#: Keyboard-ish actions offered by name in configuration.
#:
#: ``(id, kind, keys, label, repeatable, cooldown)``. Repeatability is declared
#: per key rather than globally because it is a property of the key, not of the
#: interface: holding an arrow gesture to scroll through slides is natural and
#: mirrors OS key-repeat, while holding a fist to hammer the space bar is not.
#:
#: ``{primary}`` in a key list is substituted with the platform's primary
#: modifier (Ctrl on Windows and Linux, Cmd on macOS) so a "next tab" gesture
#: means the same thing everywhere. ``Key.PRIMARY_MODIFIER`` is resolved at build
#: time in :func:`build_keyboard_actions`.
KEY_ACTION_SPECS: tuple[tuple[str, str, tuple[str, ...], str, bool, float], ...] = (
    # -- navigation -------------------------------------------------------
    ("key_space", "key", ("space",), "Space", False, 0.3),
    # Navigation keys repeat; 0.15 s is close to the OS key-repeat feel.
    ("key_left", "key", ("left",), "Left arrow", True, 0.15),
    ("key_right", "key", ("right",), "Right arrow", True, 0.15),
    ("key_up", "key", ("up",), "Up arrow", True, 0.15),
    ("key_down", "key", ("down",), "Down arrow", True, 0.15),
    ("key_page_up", "key", ("page_up",), "Page up", True, 0.25),
    ("key_page_down", "key", ("page_down",), "Page down", True, 0.25),
    ("key_home", "key", ("home",), "Home", True, 0.4),
    ("key_end", "key", ("end",), "End", True, 0.4),
    ("key_enter", "key", ("enter",), "Enter", False, 0.4),
    ("key_escape", "key", ("esc",), "Escape", False, 0.5),
    ("key_backspace", "key", ("backspace",), "Backspace", True, 0.15),
    ("key_delete", "key", ("delete",), "Delete", True, 0.25),
    ("key_tab", "key", ("tab",), "Tab", True, 0.3),
    # -- browser and document ---------------------------------------------
    ("next_tab", "key", ("{primary}", "tab"), "Next tab", False, 0.5),
    ("prev_tab", "key", ("{primary}", "shift", "tab"), "Previous tab", False, 0.5),
    ("key_back", "key", ("alt", "left"), "Back", True, 0.5),
    ("key_forward", "key", ("alt", "right"), "Forward", True, 0.5),
    # Browsers accept Ctrl+= for zoom in; the shifted '+' needs a layout-dependent
    # shift press, so the unshifted keys are used instead.
    ("key_zoom_in", "key", ("{primary}", "="), "Zoom in", True, 0.2),
    ("key_zoom_out", "key", ("{primary}", "-"), "Zoom out", True, 0.2),
    ("key_reload", "key", ("{primary}", "r"), "Reload", False, 0.8),
    # -- windows and desktop ----------------------------------------------
    ("key_show_desktop", "key", ("cmd", "d"), "Show desktop", False, 1.0),
    ("key_task_view", "key", ("cmd", "tab"), "Task view", False, 1.0),
    ("key_alt_tab", "key", ("alt", "tab"), "Switch window", False, 0.8),
    ("key_fullscreen", "key", ("f11",), "Full screen (F11)", False, 0.6),
    ("key_screenshot", "key", ("print_screen",), "Screenshot", False, 1.0),
    # -- presentation ------------------------------------------------------
    ("key_start_slideshow", "key", ("f5",), "Start slideshow (F5)", False, 1.2),
    ("key_blank_screen", "key", ("b",), "Blank screen (B)", False, 0.8),
    # -- media transport ---------------------------------------------------
    ("key_play_pause", "media", ("media_play_pause",), "Play/Pause", False, 0.6),
    ("next_track", "media", ("media_next",), "Next track", False, 0.8),
    ("previous_track", "media", ("media_previous",), "Previous track", False, 0.8),
    ("stop_media", "media", ("media_stop",), "Stop", False, 0.8),
    # Media-key volume works even when the pycaw backend does not, which makes it
    # a useful fallback on machines with no working system volume interface.
    ("media_volume_up", "media", ("media_volume_up",), "Volume + (media key)", True, 0.12),
    ("media_volume_down", "media", ("media_volume_down",), "Volume - (media key)", True, 0.12),
    ("media_volume_mute", "media", ("media_volume_mute",), "Mute (media key)", False, 0.6),
)


def primary_modifier_name() -> str:
    """Name of the platform's primary modifier key for pynput."""
    return "cmd" if sys.platform == "darwin" else "ctrl"


def build_keyboard_actions(config: ActionsConfig) -> list[Action]:
    """Create one action per entry in :data:`KEY_ACTION_SPECS`."""
    primary = primary_modifier_name()
    actions: list[Action] = []
    for action_id, kind, keys, label, repeatable, cooldown in KEY_ACTION_SPECS:
        resolved = tuple(primary if key == "{primary}" else key for key in keys)
        if kind == "key":
            actions.append(
                KeyPressAction(
                    action_id,
                    resolved,
                    label=label,
                    repeatable=repeatable,
                    cooldown=cooldown,
                    enabled=config.allow_keyboard,
                    require_focus_title=config.require_focus_title,
                )
            )
        else:
            actions.append(
                MediaKeyAction(
                    action_id,
                    resolved[0],
                    label=label,
                    repeatable=repeatable,
                    cooldown=cooldown,
                    enabled=config.allow_keyboard,
                    require_focus_title=config.require_focus_title,
                )
            )
    return actions
