"""Concrete actions: volume, mute, and screen brightness."""

from __future__ import annotations

import logging

from ..config import ActionsConfig
from .backends import (
    BackendCapability,
    BrightnessBackend,
    VolumeBackend,
    select_brightness_backend,
    select_volume_backend,
)
from .base import Action, ActionOutcome

__all__ = ["VolumeAction", "MuteAction", "BrightnessAction", "build_system_actions"]

_LOG = logging.getLogger(__name__)


class _LevelAction(Action):
    """Shared behaviour for actions that nudge a 0-100 style level."""

    def __init__(self, backend, step: float, *, lower_bound: float, upper_bound: float) -> None:
        super().__init__()
        self.backend = backend
        self.step = step
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def unavailable_reason(self) -> str:
        return self.backend.capability.reason or self.backend.capability.describe()

    def _clamp(self, value: float) -> float:
        return max(self.lower_bound, min(self.upper_bound, value))

    def describe(self) -> str:
        state = "available" if self.backend.available else f"unavailable ({self.unavailable_reason})"
        return f"{self.id}: {self.label} via {self.backend.name} [{state}]"

    def _unavailable(self) -> ActionOutcome:
        """The outcome for an expected "this machine cannot do that" condition.

        Returned rather than raised: ``Action.invoke`` is documented to report
        what happened, and a missing brightness interface is a fact about the
        machine, not an unexpected error. Genuine bugs still propagate and are
        contained by the registry.
        """
        return ActionOutcome(
            ok=False,
            message=f"{self.label} is unavailable on this system",
            detail=self.unavailable_reason,
        )

    def _require_backend(self) -> ActionOutcome | None:
        return None if self.backend.available else self._unavailable()


class VolumeAction(_LevelAction):
    """Raise or lower the system volume by a fixed step."""

    repeatable = True
    cooldown = 0.0

    def __init__(self, backend: VolumeBackend, step: float = 0.05, *, direction: int = 1) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        super().__init__(backend, step, lower_bound=0.0, upper_bound=1.0)
        self.direction = direction
        self.id = "volume_up" if direction > 0 else "volume_down"
        self.label = "Volume +" if direction > 0 else "Volume -"

    def invoke(self) -> ActionOutcome:
        unavailable = self._require_backend()
        if unavailable is not None:
            return unavailable
        current = self.backend.get_percent()
        if current is None:
            return ActionOutcome(ok=False, message="could not read the current volume")
        target = self._clamp(current + self.step * self.direction)
        if abs(target - current) < 1e-9:
            limit = "maximum" if self.direction > 0 else "minimum"
            return ActionOutcome(
                ok=True,
                message=f"already at {limit} ({current:.0%})",
                value=current,
                detail=f"backend={self.backend.name}",
            )
        self.backend.set_percent(target)
        return ActionOutcome(
            ok=True,
            message=f"{self.label}: {current:.0%} -> {target:.0%}",
            value=target,
            detail=f"backend={self.backend.name}",
        )


class MuteAction(Action):
    """Toggle (or force) the system mute state."""

    id = "mute_toggle"
    label = "Mute"
    repeatable = False
    cooldown = 0.6

    def __init__(self, backend: VolumeBackend) -> None:
        super().__init__()
        self.backend = backend

    def describe(self) -> str:
        state = "available" if self.backend.available else "unavailable"
        return f"{self.id}: {self.label} via {self.backend.name} [{state}]"

    def invoke(self) -> ActionOutcome:
        if not self.backend.available:
            return ActionOutcome(
                ok=False,
                message="Mute is unavailable on this system",
                detail=self.backend.capability.reason,
            )
        if not self.backend.supports_mute:
            return ActionOutcome(
                ok=False,
                message=f"{self.backend.name} cannot control mute",
            )
        current = self.backend.get_muted()
        target = not current if current is not None else True
        self.backend.set_muted(target)
        return ActionOutcome(
            ok=True,
            message=f"Mute {'on' if target else 'off'}",
            detail=f"backend={self.backend.name}",
        )


class BrightnessAction(_LevelAction):
    """Raise or lower screen brightness by a fixed number of percent."""

    repeatable = True
    cooldown = 0.0

    def __init__(
        self,
        backend: BrightnessBackend,
        step: int = 8,
        *,
        direction: int = 1,
        display: int | None = None,
    ) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        super().__init__(backend, float(step), lower_bound=0.0, upper_bound=100.0)
        self.direction = direction
        self.display = display
        self.id = "brightness_up" if direction > 0 else "brightness_down"
        self.label = "Brightness +" if direction > 0 else "Brightness -"

    def invoke(self) -> ActionOutcome:
        unavailable = self._require_backend()
        if unavailable is not None:
            return unavailable
        current = self.backend.get_percent()
        if current is None:
            return ActionOutcome(ok=False, message="could not read the current brightness")
        target = self._clamp(current + self.step * self.direction)
        if abs(target - current) < 1e-9:
            limit = "maximum" if self.direction > 0 else "minimum"
            return ActionOutcome(
                ok=True,
                message=f"already at {limit} ({current:.0f}%)",
                value=current,
                detail=f"backend={self.backend.name}",
            )
        self.backend.set_percent(target)
        return ActionOutcome(
            ok=True,
            message=f"{self.label}: {current:.0f}% -> {target:.0f}%",
            value=target,
            detail=f"backend={self.backend.name}",
        )


def build_system_actions(config: ActionsConfig) -> tuple[list[Action], dict[str, list[BackendCapability]]]:
    """Create volume/mute/brightness actions and report probe results.

    Returns:
        The actions, plus a mapping of subsystem name to the capability of every
        backend probed -- used by ``--doctor`` to explain what is and is not
        possible on this machine.
    """
    volume_backend, volume_capabilities = select_volume_backend()
    brightness_backend, brightness_capabilities = select_brightness_backend(
        display=config.brightness_display
    )

    actions: list[Action] = [
        VolumeAction(volume_backend, config.volume_step, direction=1),
        VolumeAction(volume_backend, config.volume_step, direction=-1),
        MuteAction(volume_backend),
        BrightnessAction(
            brightness_backend,
            config.brightness_step,
            direction=1,
            display=config.brightness_display,
        ),
        BrightnessAction(
            brightness_backend,
            config.brightness_step,
            direction=-1,
            display=config.brightness_display,
        ),
    ]

    if not volume_backend.available:
        _LOG.warning("volume control is unavailable: %s", volume_backend.capability.describe())
    if not brightness_backend.available:
        _LOG.warning(
            "brightness control is unavailable: %s", brightness_backend.capability.describe()
        )

    return actions, {
        "volume": volume_capabilities,
        "brightness": brightness_capabilities,
    }
