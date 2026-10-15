"""System backends for volume and screen brightness, with capability probing.

The original script assumed its two system integrations worked:

* ``screen_brightness_control`` -- on the machine this was refactored on, *every*
  sbc backend fails. Its Windows path calls ``wmi.WMI(namespace='wmi')`` and
  retries inside a ``@contextmanager`` that yields twice, so the real error
  ("``WmiMonitorBrightness`` does not exist") is replaced by
  ``RuntimeError: generator didn't stop after throw()``. The user sees a cryptic
  message and a brightness gesture that never works.
* ``pycaw`` -- version 20181226 (the one here) returns a bare ``IMMDevice`` from
  ``GetSpeakers()``, while newer releases return an object with an
  ``EndpointVolume`` property. Code written against one shape silently fails on
  the other.

So: never assume. Probe each backend for *read* capability at startup, in
priority order, keep the first that works, and record why the others did not.
If nothing works the action is reported as unsupported instead of failing once
per gesture with an opaque error.

Probing is strictly read-only by design. ``GetMonitorBrightness`` and friends
are safe to call; ``SetMonitorBrightness`` is only ever reached after a
successful read, so a capability check can never change the user's screen.
"""

from __future__ import annotations

import ctypes
import logging
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "BackendCapability",
    "VolumeBackend",
    "BrightnessBackend",
    "NullVolumeBackend",
    "NullBrightnessBackend",
    "PycawVolumeBackend",
    "SbcBrightnessBackend",
    "Dxva2BrightnessBackend",
    "WmiBrightnessBackend",
    "select_volume_backend",
    "select_brightness_backend",
    "reset_probe_cache",
    "is_windows",
]

_LOG = logging.getLogger(__name__)

WINDOWS = sys.platform == "win32"


def is_windows() -> bool:
    return WINDOWS


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """The outcome of probing one backend."""

    name: str
    available: bool
    detail: str = ""
    reason: str = ""

    def describe(self) -> str:
        status = "ok" if self.available else "unavailable"
        extra = self.detail or self.reason
        return f"{self.name}: {status}" + (f" ({extra})" if extra else "")


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #


class VolumeBackend(ABC):
    """Reads and writes the system output volume."""

    name = "volume"
    supports_mute = False

    def __init__(self) -> None:
        self._capability: BackendCapability | None = None

    @property
    def capability(self) -> BackendCapability:
        if self._capability is None:
            try:
                self._capability = self.probe()
            except Exception as exc:  # pragma: no cover - defensive
                self._capability = BackendCapability(
                    self.name, False, reason=f"{type(exc).__name__}: {exc}"
                )
        return self._capability

    @property
    def available(self) -> bool:
        return self.capability.available

    @abstractmethod
    def probe(self) -> BackendCapability:
        """Check availability using read-only calls only."""

    @abstractmethod
    def get_percent(self) -> float | None:
        """Current volume as 0-1, or ``None`` if it cannot be read."""

    @abstractmethod
    def set_percent(self, value: float) -> None:
        """Set volume; ``value`` is clamped to 0-1."""

    def get_muted(self) -> bool | None:
        return None

    def set_muted(self, muted: bool) -> None:
        raise NotImplementedError(f"{self.name} cannot mute")

    def describe(self) -> str:
        return self.capability.describe()


class NullVolumeBackend(VolumeBackend):
    """Placeholder used when no real backend is available."""

    name = "none"

    def __init__(self, reason: str = "no supported volume interface was found") -> None:
        super().__init__()
        self._reason = reason

    def probe(self) -> BackendCapability:
        return BackendCapability(self.name, False, reason=self._reason)

    def get_percent(self) -> float | None:
        return None

    def set_percent(self, value: float) -> None:
        raise RuntimeError(f"volume is not controllable: {self._reason}")


class PycawVolumeBackend(VolumeBackend):
    """Windows volume through ``pycaw``.

    Handles both pycaw API generations. The older release (20181226, the version
    on the machine this was written against) hands back a raw ``IMMDevice`` that
    must be activated and cast to ``IAudioEndpointVolume``; newer releases expose
    ``AudioDevice.EndpointVolume`` directly.
    """

    name = "pycaw"
    supports_mute = True

    def __init__(self) -> None:
        super().__init__()
        self._endpoint = None
        self._comtypes = None

    def _connect(self):
        if self._endpoint is not None:
            return self._endpoint

        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        device = AudioUtilities.GetSpeakers()

        # Newer pycaw: the device already exposes the endpoint volume interface.
        endpoint = getattr(device, "EndpointVolume", None)
        if endpoint is None:
            # Older pycaw: activate the interface on the raw IMMDevice.
            interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            endpoint = cast(interface, POINTER(IAudioEndpointVolume))

        self._endpoint = endpoint
        return endpoint

    def reconnect(self) -> None:
        """Drop the cached interface; call after the default device changes."""
        self._endpoint = None

    def probe(self) -> BackendCapability:
        if not WINDOWS:
            return BackendCapability(self.name, False, reason="Windows only")
        try:
            import pycaw  # noqa: F401
        except Exception as exc:
            return BackendCapability(self.name, False, reason=f"pycaw not importable: {exc}")
        try:
            endpoint = self._connect()
            level = float(endpoint.GetMasterVolumeLevelScalar())
        except Exception as exc:
            return BackendCapability(
                self.name, False, reason=f"{type(exc).__name__}: {exc}"[:200]
            )
        return BackendCapability(self.name, True, detail=f"master volume {level:.0%}")

    def get_percent(self) -> float | None:
        try:
            return float(self._connect().GetMasterVolumeLevelScalar())
        except Exception as exc:
            self.reconnect()
            _LOG.warning("could not read volume: %s", exc)
            return None

    def set_percent(self, value: float) -> None:
        target = max(0.0, min(1.0, float(value)))
        try:
            self._connect().SetMasterVolumeLevelScalar(target, None)
        except Exception:
            # The default endpoint can change (headphones plugged in); retry once
            # against a fresh interface before giving up.
            self.reconnect()
            self._connect().SetMasterVolumeLevelScalar(target, None)

    def get_muted(self) -> bool | None:
        try:
            return bool(self._connect().GetMute())
        except Exception as exc:
            _LOG.warning("could not read mute state: %s", exc)
            return None

    def set_muted(self, muted: bool) -> None:
        self._connect().SetMute(1 if muted else 0, None)

    def describe(self) -> str:
        return f"{self.name} ({self.capability.detail or 'ok'})"


# --------------------------------------------------------------------------- #
# Brightness
# --------------------------------------------------------------------------- #


class BrightnessBackend(ABC):
    """Reads and writes screen brightness as a percentage."""

    name = "brightness"

    def __init__(self) -> None:
        self._capability: BackendCapability | None = None

    @property
    def capability(self) -> BackendCapability:
        if self._capability is None:
            try:
                self._capability = self.probe()
            except Exception as exc:  # pragma: no cover - defensive
                self._capability = BackendCapability(
                    self.name, False, reason=f"{type(exc).__name__}: {exc}"
                )
        return self._capability

    @property
    def available(self) -> bool:
        return self.capability.available

    @abstractmethod
    def probe(self) -> BackendCapability:
        """Check availability using read-only calls only."""

    @abstractmethod
    def get_percent(self) -> float | None:
        """Current brightness 0-100, or ``None``."""

    @abstractmethod
    def set_percent(self, value: float) -> None:
        """Set brightness; ``value`` is clamped to 0-100."""

    def describe(self) -> str:
        return self.capability.describe()


class NullBrightnessBackend(BrightnessBackend):
    """Placeholder used when no real backend is available."""

    name = "none"

    def __init__(self, reason: str = "no supported brightness interface was found") -> None:
        super().__init__()
        self._reason = reason

    def probe(self) -> BackendCapability:
        return BackendCapability(self.name, False, reason=self._reason)

    def get_percent(self) -> float | None:
        return None

    def set_percent(self, value: float) -> None:
        raise RuntimeError(f"brightness is not controllable: {self._reason}")


class SbcBrightnessBackend(BrightnessBackend):
    """Brightness through ``screen_brightness_control``.

    Works well on most laptops, but its Windows path is fragile: it needs the
    ``wmi`` package and a ``WmiMonitorBrightness`` WMI class that some machines
    and remote sessions do not expose. When it breaks it raises
    ``RuntimeError: generator didn't stop after throw()``, which is what the
    probe reports verbatim so the cause is at least visible.
    """

    name = "screen_brightness_control"

    def __init__(self, display: int | None = None) -> None:
        super().__init__()
        self.display = display
        self._module = None

    def _module_or_raise(self):
        if self._module is None:
            import screen_brightness_control as sbc

            self._module = sbc
        return self._module

    def probe(self) -> BackendCapability:
        try:
            module = self._module_or_raise()
        except Exception as exc:
            return BackendCapability(self.name, False, reason=f"not importable: {exc}")
        try:
            values = module.get_brightness(display=self.display)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if "didn't stop after throw" in str(exc):
                reason += (
                    " -- this is an sbc bug that masks the real cause; try "
                    "'pip install wmi', or use the wmi backend"
                )
            return BackendCapability(self.name, False, reason=reason[:240])

        reading = next((v for v in values if v is not None), None)
        if reading is None:
            return BackendCapability(self.name, False, reason="reported no readable displays")
        return BackendCapability(self.name, True, detail=f"{reading}% ({len(values)} display(s))")

    def get_percent(self) -> float | None:
        values = self._module_or_raise().get_brightness(display=self.display)
        reading = next((v for v in values if v is not None), None)
        return float(reading) if reading is not None else None

    def set_percent(self, value: float) -> None:
        self._module_or_raise().set_brightness(
            int(round(max(0.0, min(100.0, float(value))))), display=self.display, force=True
        )


class Dxva2BrightnessBackend(BrightnessBackend):
    """Brightness via the Windows DDC/CI API in ``dxva2.dll``.

    No third-party dependency, and it works on many external monitors where the
    WMI class does not exist. Requires a *successful read* before writes are
    allowed: some drivers return success from ``SetMonitorBrightness`` while
    ``GetMonitorBrightness`` fails, and writing blind to a display that cannot
    report its state is not something a gesture should do.
    """

    name = "dxva2"

    class _PHYSICAL_MONITOR(ctypes.Structure):
        _fields_ = [
            ("hPhysicalMonitor", ctypes.c_void_p),
            ("szPhysicalMonitorDescription", ctypes.c_wchar * 128),
        ]

    def __init__(self) -> None:
        super().__init__()
        self._handle = None
        self._bounds: tuple[int, int] | None = None

    def _enumerate(self) -> list:
        """Physical monitors behind every display monitor."""
        if not WINDOWS:
            return []
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32.dll")
        dxva2 = ctypes.WinDLL("dxva2.dll")

        handles: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
            ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
        )

        def collect(monitor, _hdc, _rect, _data):
            handles.append(monitor)
            return True

        user32.EnumDisplayMonitors(None, None, callback_type(collect), 0)

        found = []
        for handle in handles:
            count = wintypes.DWORD()
            if not dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR(handle, ctypes.byref(count)):
                continue
            array = (self._PHYSICAL_MONITOR * count.value)()
            if not dxva2.GetPhysicalMonitorsFromHMONITOR(handle, count.value, array):
                continue
            found.extend(array)
        return found

    def _read_brightness(self, handle) -> tuple[bool, int, int, int]:
        """Read min/max/current for a physical monitor handle. Read-only."""
        dxva2 = ctypes.WinDLL("dxva2.dll")
        minimum, maximum, current = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
        ok = dxva2.GetMonitorBrightness(
            handle, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(current)
        )
        return bool(ok), int(minimum.value), int(maximum.value), int(current.value)

    def probe(self) -> BackendCapability:
        if not WINDOWS:
            return BackendCapability(self.name, False, reason="Windows only")
        try:
            monitors = self._enumerate()
        except Exception as exc:
            return BackendCapability(self.name, False, reason=f"{type(exc).__name__}: {exc}")
        if not monitors:
            return BackendCapability(self.name, False, reason="no physical monitors found")

        for monitor in monitors:
            try:
                ok, minimum, maximum, current = self._read_brightness(monitor.hPhysicalMonitor)
            except Exception:
                continue
            if ok and maximum > minimum:
                self._handle = monitor.hPhysicalMonitor
                self._bounds = (minimum, maximum)
                percent = (current - minimum) / float(maximum - minimum) * 100.0
                return BackendCapability(
                    self.name, True, detail=f"{percent:.0f}% (DDC/CI supported)"
                )
        return BackendCapability(
            self.name,
            False,
            reason=(
                "the display does not report brightness over DDC/CI; common for "
                "internal panels and remote sessions"
            ),
        )

    def _percent_to_raw(self, value: float) -> int:
        assert self._bounds is not None  # noqa: S101 - guarded by availability
        minimum, maximum = self._bounds
        return int(round(minimum + (maximum - minimum) * max(0.0, min(100.0, value)) / 100.0))

    def get_percent(self) -> float | None:
        if self._handle is None or self._bounds is None:
            return None
        ok, minimum, maximum, current = self._read_brightness(self._handle)
        if not ok or maximum <= minimum:
            return None
        return (current - minimum) / float(maximum - minimum) * 100.0

    def set_percent(self, value: float) -> None:
        if self._handle is None or self._bounds is None:
            raise RuntimeError("no DDC/CI-capable display was found at startup")
        dxva2 = ctypes.WinDLL("dxva2.dll")
        if not dxva2.SetMonitorBrightness(self._handle, self._percent_to_raw(value)):
            raise RuntimeError("SetMonitorBrightness was rejected by the display driver")


#: WMI status codes worth explaining rather than echoing as a hex number.
_WMI_HRESULT_HINTS = {
    "0x80041002": "the WmiMonitorBrightness class does not exist on this system",
    "0x80041003": (
        "WMI access was denied; the class may be absent, or WMI permissions may "
        "restrict it (an elevated shell sometimes helps)"
    ),
    "0x80041010": "the WMI class name was rejected by the provider",
    "0x80070490": "no monitor provider reports brightness on this system",
}


def _explain_wmi_error(stderr: str) -> str:
    """Turn PowerShell/WMI noise into one useful line.

    A bare ``HRESULT 0x80041003`` tells the user nothing, and guessing a single
    cause would be worse than saying what is actually known.
    """
    text = (stderr or "").strip()
    if not text:
        return "no output from the WMI query"
    for code, hint in _WMI_HRESULT_HINTS.items():
        if code.lower() in text.lower():
            return f"WMI reported {code}: {hint}"
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and not line.startswith("+") and "CategoryInfo" not in line:
            return line[:160]
    return text.splitlines()[-1].strip()[:160]


class WmiBrightnessBackend(BrightnessBackend):
    """Brightness through the ``WmiMonitorBrightness`` WMI class.

    The standard route for internal laptop panels. Run through PowerShell so it
    needs no Python dependency at all. Each call costs a few hundred
    milliseconds, which is why actions are executed off the render thread (see
    :class:`~gesture_control.actions.base.ActionExecutor`).

    Some systems genuinely lack this class -- remote desktop sessions commonly
    do -- in which case the probe reports it and the gesture is disabled rather
    than failing once per gesture with an opaque error.
    """

    name = "wmi"

    _GET = (
        "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness "
        "-ErrorAction Stop | Select-Object -First 1 -ExpandProperty CurrentBrightness)"
    )
    _SET = (
        "$m = Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods "
        "-ErrorAction Stop | Select-Object -First 1; "
        "if ($null -eq $m) { exit 3 }; $m.WmiSetBrightness(1, {value})"
    )

    def __init__(self, timeout: float = 8.0) -> None:
        super().__init__()
        self.timeout = timeout

    def _powershell(self) -> Sequence[str] | None:
        for candidate in ("powershell", "pwsh"):
            path = shutil.which(candidate)
            if path:
                return (path, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-Command")
        return None

    def _run(self, script: str) -> subprocess.CompletedProcess | None:
        shell = self._powershell()
        if shell is None:
            return None
        try:
            return subprocess.run(
                [*shell, script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            _LOG.debug("powershell call failed: %s", exc)
            return None

    def probe(self) -> BackendCapability:
        if not WINDOWS:
            return BackendCapability(self.name, False, reason="Windows only")
        if self._powershell() is None:
            return BackendCapability(self.name, False, reason="PowerShell not found on PATH")
        result = self._run(self._GET)
        if result is None:
            return BackendCapability(self.name, False, reason="PowerShell call timed out")
        text = (result.stdout or "").strip()
        if not text:
            return BackendCapability(
                self.name,
                False,
                reason=_explain_wmi_error(result.stderr or ""),
            )
        try:
            value = int(float(text))
        except ValueError:
            return BackendCapability(self.name, False, reason=f"unexpected output {text[:60]!r}")
        return BackendCapability(self.name, True, detail=f"{value}%")

    def get_percent(self) -> float | None:
        result = self._run(self._GET)
        if result is None:
            return None
        try:
            return float(int(float((result.stdout or "").strip())))
        except (ValueError, TypeError):
            return None

    def set_percent(self, value: float) -> None:
        target = int(round(max(0.0, min(100.0, float(value)))))
        result = self._run(self._SET.format(value=target))
        if result is None or result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()[-1:] if result else []
            raise RuntimeError(
                f"WmiSetBrightness failed" + (f": {detail[0][:160]}" if detail else "")
            )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

_VOLUME_CACHE: dict[str, tuple[VolumeBackend, list[BackendCapability]]] = {}
_BRIGHTNESS_CACHE: dict[str, tuple[BrightnessBackend, list[BackendCapability]]] = {}


def reset_probe_cache() -> None:
    """Forget cached probe results. Used by the tests."""
    _VOLUME_CACHE.clear()
    _BRIGHTNESS_CACHE.clear()


def _select(
    backends: Iterable,
    preferred: str | None,
    cache: dict,
    key: str,
    null_factory,
) -> tuple:
    """Probe backends in order, returning the first usable one plus all results."""
    if preferred is None and key in cache:
        return cache[key]

    ordered = list(backends)
    capabilities: list[BackendCapability] = []
    chosen = None

    for backend in ordered:
        capability = backend.capability
        capabilities.append(capability)
        if capability.available and chosen is None:
            if preferred is None or backend.name == preferred:
                chosen = backend

    if chosen is None:
        # Keep this short: it ends up in log lines and on the HUD. The full
        # per-backend explanation is returned separately in ``capabilities`` and
        # surfaced by --doctor.
        prefix = (
            f"requested backend {preferred!r} is unavailable"
            if preferred is not None
            else "no working backend found"
        )
        chosen = null_factory(f"{prefix} (run --doctor for details)")

    result = (chosen, capabilities)
    if preferred is None:
        cache[key] = result
    return result


def select_volume_backend(preferred: str | None = None) -> tuple[VolumeBackend, list[BackendCapability]]:
    """Pick the best available volume backend, probing at most once per process."""
    return _select(
        [PycawVolumeBackend()],
        preferred,
        _VOLUME_CACHE,
        "volume",
        NullVolumeBackend,
    )


def select_brightness_backend(
    preferred: str | None = None, *, display: int | None = None
) -> tuple[BrightnessBackend, list[BackendCapability]]:
    """Pick the best available brightness backend.

    Order is deliberate: ``screen_brightness_control`` is tried first because it
    is the most capable when it works, then the dependency-free DDC/CI path, then
    WMI. The null backend is always last so there is always an answer.
    """
    return _select(
        [
            SbcBrightnessBackend(display),
            Dxva2BrightnessBackend(),
            WmiBrightnessBackend(),
        ],
        preferred,
        _BRIGHTNESS_CACHE,
        "brightness",
        NullBrightnessBackend,
    )
