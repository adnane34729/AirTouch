"""Diagnostics: ``--doctor`` and ``--self-test``.

Two questions this module answers, both of which the original could not:

``--doctor`` -- *why is this not working on my machine?*
    Checks the interpreter, each dependency, the camera, the detector, and every
    volume and brightness backend, and reports the actual reason each one failed.
    On a machine with several Python installations the usual cause of "it just
    crashes" is that the dependencies are installed in a *different* interpreter
    than the one running the script, so the interpreter path is the first check.

``--self-test`` -- *is the recogniser actually correct?*
    Drives every configured gesture through the real pose and classification
    pipeline using the synthetic hand model, at several rotations and hand sizes,
    and reports pass/fail per gesture. No camera, no screen, no hand needed. This
    is the same machinery the unit tests use, exposed so it can be run against
    the user's own configuration.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from typing import Sequence

from . import __version__
from .config import AppConfig, ConfigError
from .gestures import Requirement
from .synthetic import make_hand
from .types import FINGERS, Finger

__all__ = [
    "Check",
    "CaseResult",
    "run_doctor",
    "format_doctor",
    "run_self_test",
    "format_self_test",
    "SUPPORTED_PYTHON",
]

#: MediaPipe publishes wheels for these versions; anything outside is a warning.
SUPPORTED_PYTHON = "3.9 - 3.12"

_OK = "ok"
_WARN = "warn"
_FAIL = "fail"

_SYMBOLS = {_OK: "PASS", _WARN: "WARN", _FAIL: "FAIL"}


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic finding."""

    name: str
    status: str
    detail: str = ""
    advice: str = ""

    @property
    def ok(self) -> bool:
        return self.status == _OK


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One self-test case."""

    gesture: str
    expected: str
    actual: str
    rotation: float
    scale: float
    ok: bool
    detail: str = ""


def _dependency_check(module: str, *, required: bool, purpose: str) -> Check:
    try:
        imported = __import__(module)
    except Exception as exc:
        return Check(
            name=f"dependency: {module}",
            status=_FAIL if required else _WARN,
            detail=f"not importable ({type(exc).__name__}: {exc})",
            advice=(
                f"install it for this interpreter:  "
                f'"{sys.executable}" -m pip install -r requirements.txt   ({purpose})'
            ),
        )
    version = getattr(imported, "__version__", "")
    return Check(
        name=f"dependency: {module}",
        status=_OK,
        detail=f"{version or 'installed'} ({purpose})",
    )


def _interpreter_check() -> Check:
    version = sys.version_info
    detail = f"Python {platform.python_version()} at {sys.executable}"
    if version[:2] < (3, 9):
        return Check(
            "interpreter",
            _FAIL,
            detail,
            f"Python {SUPPORTED_PYTHON} is required",
        )
    if version[:2] > (3, 12):
        return Check(
            "interpreter",
            _WARN,
            detail,
            f"MediaPipe has no wheels for this version yet; use Python {SUPPORTED_PYTHON}. "
            f"See run.py, which picks a suitable interpreter automatically.",
        )
    return Check("interpreter", _OK, detail)


def _camera_check(config: AppConfig) -> Check:
    try:
        import cv2
    except Exception as exc:
        return Check("camera", _FAIL, f"OpenCV unavailable ({exc})")

    started = time.perf_counter()
    capture = None
    try:
        capture = cv2.VideoCapture(config.camera.index)
        if not capture.isOpened():
            return Check(
                "camera",
                _FAIL,
                f"camera index {config.camera.index} could not be opened",
                "close other apps using the camera, allow camera access in "
                "Windows privacy settings, or try --camera 1",
            )
        ok, frame = capture.read()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not ok or frame is None:
            return Check(
                "camera",
                _WARN,
                f"opened but returned no frame ({width}x{height})",
                "another application may be holding the device",
            )
        shape = frame.shape[:2]
        elapsed = (time.perf_counter() - started) * 1000
        return Check(
            "camera",
            _OK,
            f"index {config.camera.index}: {width}x{height} @ {fps:.0f} fps, "
            f"first frame {shape[1]}x{shape[0]} in {elapsed:.0f} ms",
        )
    except Exception as exc:
        return Check("camera", _FAIL, f"{type(exc).__name__}: {exc}")
    finally:
        if capture is not None:
            capture.release()


def _detector_check(config: AppConfig) -> Check:
    from .detector import DetectorError, create_detector, mediapipe_available

    if not mediapipe_available():
        return Check(
            "hand detector",
            _FAIL,
            "MediaPipe is not importable in this interpreter",
            f'install it:  "{sys.executable}" -m pip install -r requirements.txt',
        )
    detector = None
    try:
        detector = create_detector("mediapipe", config.detector)
        import numpy as np

        frame = np.zeros((config.camera.height or 720, config.camera.width or 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0.0)  # warm up the TFLite delegate
        started = time.perf_counter()
        detector.detect(frame, 0.01)
        elapsed = (time.perf_counter() - started) * 1000
        return Check(
            "hand detector",
            _OK,
            f"{detector.describe()}; ~{elapsed:.0f} ms per frame "
            f"(budget {(1000 / elapsed):.0f} fps on this CPU)",
        )
    except DetectorError as exc:
        return Check("hand detector", _FAIL, str(exc).splitlines()[0])
    except Exception as exc:
        return Check("hand detector", _FAIL, f"{type(exc).__name__}: {exc}")
    finally:
        if detector is not None:
            detector.close()


def _volume_checks(config: AppConfig) -> list[Check]:
    from .actions.backends import select_volume_backend

    backend, capabilities = select_volume_backend()
    checks = [
        Check(
            f"volume backend: {capability.name}",
            _OK if capability.available else _WARN,
            capability.detail or capability.reason,
        )
        for capability in capabilities
    ]
    if backend.available:
        level = backend.get_percent()
        muted = backend.get_muted()
        detail = f"using {backend.name}"
        if level is not None:
            detail += f"; master volume {level:.0%}"
        if muted is not None:
            detail += f"; muted={muted}"
        checks.append(Check("volume control", _OK, detail))
    else:
        checks.append(
            Check(
                "volume control",
                _FAIL,
                backend.capability.reason,
                "volume gestures will report an error; on Windows install "
                "pycaw and comtypes (pip install pycaw comtypes)",
            )
        )
    return checks


def _brightness_checks(config: AppConfig) -> list[Check]:
    from .actions.backends import select_brightness_backend

    backend, capabilities = select_brightness_backend(display=config.actions.brightness_display)
    checks = [
        Check(
            f"brightness backend: {capability.name}",
            _OK if capability.available else _WARN,
            capability.detail or capability.reason,
        )
        for capability in capabilities
    ]
    if backend.available:
        level = backend.get_percent()
        checks.append(
            Check(
                "brightness control",
                _OK,
                f"using {backend.name}"
                + (f"; current brightness {level:.0f}%" if level is not None else ""),
            )
        )
    else:
        checks.append(
            Check(
                "brightness control",
                _WARN,
                "no working brightness interface was found",
                "this is common on desktops, external monitors without DDC/CI, "
                "and remote sessions. Brightness gestures will report "
                "'unavailable' instead of failing opaquely. Everything else works.",
            )
        )
    return checks


def _keyboard_checks(config: AppConfig) -> list[Check]:
    from .actions.keyboard import foreground_window_title

    checks: list[Check] = []
    try:
        from pynput.keyboard import Controller, Key

        Controller()
        media = [name for name in ("media_play_pause", "media_next", "media_previous") if hasattr(Key, name)]
        checks.append(
            Check("keyboard backend", _OK, f"pynput controller created; media keys: {len(media)}/3")
        )
    except Exception as exc:
        checks.append(
            Check(
                "keyboard backend",
                _WARN,
                f"{type(exc).__name__}: {exc}",
                "keyboard and media gestures will fail; volume and brightness are unaffected",
            )
        )

    if not config.actions.allow_keyboard:
        checks.append(Check("keyboard actions", _WARN, "disabled by configuration"))
    title = foreground_window_title()
    if config.actions.require_focus_title:
        matches = title is not None and config.actions.require_focus_title.lower() in title.lower()
        checks.append(
            Check(
                "focus guard",
                _OK if matches else _WARN,
                f"requires {config.actions.require_focus_title!r}; focused window is {title!r}",
                "" if matches else "keyboard gestures will do nothing until that window is focused",
            )
        )
    elif title is not None:
        checks.append(Check("focus guard", _OK, f"disabled (focused window is {title!r})"))
    return checks


def _gesture_checks(config: AppConfig) -> list[Check]:
    from .actions import build_registry, resolve_modes, resolve_timing

    checks: list[Check] = []
    try:
        mode_set = config.build_mode_set()
        two_hand_set = config.build_two_hand_set()
    except ConfigError as exc:
        return [Check("gestures", _FAIL, str(exc))]

    total = sum(len(mode) for mode in mode_set)
    checks.append(
        Check(
            "modes",
            _OK,
            f"{len(mode_set)} mode(s) from preset {config.preset!r}, {total} gesture(s): "
            f"{', '.join(mode_set.names)}",
        )
    )
    for mode in mode_set:
        checks.append(
            Check(
                f"mode: {mode.name}",
                _OK if mode.gestures else _WARN,
                f"{mode.label}, {len(mode)} gesture(s)"
                + (f": {', '.join(s.label for s in mode.gestures)}" if mode.gestures else " (inert)"),
            )
        )
    for warning in mode_set.warnings:
        checks.append(Check("gesture overlap", _WARN, warning))
    for warning in two_hand_set.warnings:
        checks.append(Check("two-hand overlap", _WARN, warning))

    # Resolve against the actions that really exist, so a typo in a custom config
    # is caught here rather than at first gesture.
    registry, _ = build_registry(
        config.actions, mode_names=mode_set.names, async_actions=False
    )
    try:
        resolve_modes(config.modes, registry)
        resolve_timing(config.two_hand, registry, context="two-hand gesture")
    except ConfigError as exc:
        checks.append(Check("gesture actions", _FAIL, str(exc)))
        return checks

    checks.append(
        Check(
            "gesture actions",
            _OK,
            f"{len(registry)} action(s) available: {', '.join(registry.ids[:6])}"
            + (" ..." if len(registry) > 6 else ""),
        )
    )
    checks.append(
        Check(
            "two-hand gestures",
            _OK if two_hand_set.gestures else _WARN,
            f"{len(two_hand_set)} configured: "
            + (", ".join(g.label for g in two_hand_set.gestures) or "none"),
            "" if two_hand_set.gestures else
            "without them there is no gesture for mode switching, pause or quit",
        )
    )

    for mode in config.modes:
        for spec in mode.gestures:
            if spec.repeatable:
                action = registry.get(spec.action)
                if action is not None and not action.repeatable:
                    checks.append(
                        Check(
                            f"repeat: {mode.name}/{spec.name}",
                            _WARN,
                            f"gesture is repeatable but action {spec.action!r} is not; "
                            f"holding it will only act once",
                        )
                    )
    return checks


def run_doctor(config: AppConfig) -> list[Check]:
    """Run every diagnostic. Never raises: a failed check is a result."""
    checks: list[Check] = [
        Check("version", _OK, f"gesture-control {__version__}"),
        _interpreter_check(),
        _dependency_check("numpy", required=True, purpose="landmark maths"),
        _dependency_check("cv2", required=True, purpose="camera and display"),
        _dependency_check("mediapipe", required=True, purpose="hand detection"),
        _dependency_check("pynput", required=False, purpose="keyboard and media keys"),
    ]
    if sys.platform == "win32":
        checks.append(_dependency_check("comtypes", required=False, purpose="Windows audio"))
        checks.append(_dependency_check("pycaw", required=False, purpose="Windows volume"))
        checks.append(
            _dependency_check(
                "screen_brightness_control", required=False, purpose="screen brightness"
            )
        )
    checks.append(_detector_check(config))
    checks.append(_camera_check(config))
    checks.extend(_volume_checks(config))
    checks.extend(_brightness_checks(config))
    checks.extend(_keyboard_checks(config))
    checks.extend(_gesture_checks(config))
    return checks


def format_doctor(checks: Sequence[Check]) -> str:
    """Render doctor output as an aligned, readable report."""
    lines: list[str] = []
    width = max((len(check.name) for check in checks), default=10)
    for check in checks:
        symbol = _SYMBOLS.get(check.status, check.status.upper())
        lines.append(f"[{symbol}] {check.name:<{width}}  {check.detail}")
        if check.advice:
            lines.append(f"{'':<7}{'':<{width}}  -> {check.advice}")

    failures = sum(1 for check in checks if check.status == _FAIL)
    warnings = sum(1 for check in checks if check.status == _WARN)
    lines.append("")
    lines.append(
        f"{len(checks)} checks: {len(checks) - failures - warnings} passed, "
        f"{warnings} warning(s), {failures} failure(s)"
    )
    if failures:
        lines.append("Fix the FAIL lines above before running the controller.")
    elif warnings:
        lines.append("Warnings do not prevent the controller from running.")
    else:
        lines.append("Everything looks healthy.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self test
# --------------------------------------------------------------------------- #

#: Rotations (degrees) and hand sizes each gesture is verified at. Rotation is
#: the property the original's tip-versus-pip comparison could not survive.
_SELF_TEST_ROTATIONS: tuple[float, ...] = (0.0, 40.0, 90.0, 180.0, -60.0)
_SELF_TEST_SCALES: tuple[float, ...] = (0.16, 0.28, 0.42)


def _curls_for_requirements(requirements) -> tuple[dict[Finger, float], float]:
    """Build a hand shape that satisfies a requirement table.

    Fingers required UP are extended, fingers required DOWN are curled, and
    fingers the gesture ignores (ANY) are curled -- except the thumb, which is
    only held out when the gesture explicitly wants it, since an out thumb is
    what distinguishes a thumbs-up from a fist.
    """
    curls: dict[Finger, float] = {}
    for finger in FINGERS:
        requirement = requirements.get(finger, Requirement.ANY)
        curls[finger] = 0.0 if requirement is Requirement.UP else 1.0
    thumb_out = 1.0 if requirements.get(Finger.THUMB, Requirement.ANY) is Requirement.UP else 0.0
    return curls, thumb_out


def _curls_for(spec) -> tuple[dict[Finger, float], float]:
    """Backwards-compatible alias used for single-hand gestures."""
    return _curls_for_requirements(spec.requirements)


def run_self_test(config: AppConfig) -> list[CaseResult]:
    """Verify every configured gesture is recognised, in every mode, by hand count.

    Three families of case:

    * each single-hand gesture of each mode, across rotations and hand sizes;
    * each two-hand gesture, with both hands in the right shape;
    * the *negative* cases that make two-hand gestures safe -- one hand alone
      must never satisfy a two-hand gesture, and no hand at all must never fire
      anything.
    """
    from .geometry import make_observation
    from .modes import ModeSet
    from .pose import PoseEstimator

    thresholds = config.build_thresholds()
    mode_set = ModeSet(modes=config.modes, wrap=config.switching.wrap)
    two_hand_set = config.build_two_hand_set()
    results: list[CaseResult] = []

    def pose_of(curls, thumb_out, rotation, scale, mirror=False):
        from .types import Handedness as _Handedness

        raw = make_hand(
            curls,
            thumb_out=thumb_out,
            rotation_deg=rotation,
            scale=scale,
            aspect_ratio=16 / 9,
            mirror=mirror,
        )
        observation = make_observation(
            raw,
            aspect_ratio=16 / 9,
            handedness=_Handedness.LEFT if mirror else _Handedness.RIGHT,
            handedness_score=1.0,
        )
        # A fresh estimator per case: hysteresis must not be what makes a case pass.
        return PoseEstimator(thresholds).estimate(observation)

    # -- single-hand gestures, per mode ----------------------------------- #
    for mode in mode_set:
        gesture_set = mode.build_gesture_set()
        if gesture_set is None:
            continue
        for spec in gesture_set:
            curls, thumb_out = _curls_for(spec)
            for rotation in _SELF_TEST_ROTATIONS:
                for scale in _SELF_TEST_SCALES:
                    pose = pose_of(curls, thumb_out, rotation, scale)
                    match = gesture_set.classify(pose)
                    actual = match.name if match is not None else "<none>"
                    results.append(
                        CaseResult(
                            gesture=f"{mode.name}/{spec.name}",
                            expected=spec.name,
                            actual=actual,
                            rotation=rotation,
                            scale=scale,
                            ok=match is not None and match.name == spec.name,
                            detail=f"hand shape {spec.pose_text()} -> fingers {pose.signature}",
                        )
                    )

    # -- two-hand gestures ------------------------------------------------- #
    for gesture in two_hand_set:
        tables = (
            [(gesture.shape, gesture.shape)]
            if gesture.symmetric
            else [(gesture.left or {}, gesture.right or {})]
        )
        for left_table, right_table in tables:
            left_curls, left_thumb = _curls_for_requirements(left_table)
            right_curls, right_thumb = _curls_for_requirements(right_table)
            for rotation in _SELF_TEST_ROTATIONS:
                # Straight and swapped, to prove the match does not depend on the
                # detector labelling the hands correctly.
                for swap in (False, True):
                    first = pose_of(
                        left_curls, left_thumb, rotation, 0.26, mirror=swap
                    )
                    second = pose_of(
                        right_curls, right_thumb, rotation, 0.26, mirror=not swap
                    )
                    match = two_hand_set.classify(first, second)
                    actual = match.name if match is not None else "<none>"
                    results.append(
                        CaseResult(
                            gesture=f"two-hand/{gesture.name}",
                            expected=gesture.name,
                            actual=actual,
                            rotation=rotation,
                            scale=0.26,
                            ok=match is not None and match.name == gesture.name,
                            detail=f"{gesture.hand_text()} (swapped={swap})",
                        )
                    )

            # The safety property: one hand alone must not be enough.
            single = pose_of(left_curls, left_thumb, 0.0, 0.26)
            results.append(
                CaseResult(
                    gesture=f"two-hand/{gesture.name} (one hand only)",
                    expected="<none>",
                    actual=two_hand_set.classify(single, None).name
                    if two_hand_set.classify(single, None)
                    else "<none>",
                    rotation=0.0,
                    scale=0.26,
                    ok=two_hand_set.classify(single, None) is None,
                    detail="a two-hand gesture must never fire with one hand",
                )
            )

    # -- nothing at all ---------------------------------------------------- #
    results.append(
        CaseResult(
            gesture="<no hand>",
            expected="<none>",
            actual="<none>",
            rotation=0.0,
            scale=0.0,
            ok=two_hand_set.classify(None, None) is None,
            detail="no detection must never trigger a two-hand action",
        )
    )

    # An open palm must resolve to the gesture that requires all five fingers up
    # -- whatever that mode calls it -- and to nothing at all in modes that leave
    # it unmapped. What it must never do is resolve to some *other* gesture.
    all_up = {finger: Requirement.UP for finger in FINGERS}
    open_curls, open_thumb = _curls_for_requirements(all_up)
    for mode in mode_set:
        gesture_set = mode.build_gesture_set()
        if gesture_set is None:
            continue
        pose = pose_of(open_curls, open_thumb, 0.0, 0.26)
        match = gesture_set.classify(pose)
        actual = match.name if match else "<none>"
        declared = next(
            (
                spec
                for spec in gesture_set
                if all(
                    spec.requirements.get(finger, Requirement.ANY) is Requirement.UP
                    for finger in FINGERS
                )
            ),
            None,
        )
        expected = declared.name if declared is not None else "<none>"
        results.append(
            CaseResult(
                gesture=f"{mode.name}/<open hand>",
                expected=expected,
                actual=actual,
                rotation=0.0,
                scale=0.26,
                ok=actual == expected,
                detail=(
                    "an open palm must map to the all-fingers-up gesture, or to "
                    "nothing if the mode defines none"
                ),
            )
        )
    return results


def format_self_test(results: Sequence[CaseResult], *, verbose: bool = False) -> str:
    """Render self-test results as a per-gesture summary plus any failures."""
    lines: list[str] = []
    by_gesture: dict[str, list[CaseResult]] = {}
    for result in results:
        by_gesture.setdefault(result.gesture, []).append(result)

    for gesture, cases in by_gesture.items():
        passed = sum(1 for case in cases if case.ok)
        total = len(cases)
        status = "PASS" if passed == total else "FAIL"
        lines.append(f"[{status}] {gesture:<16} {passed}/{total} cases")
        for case in cases:
            if not case.ok or verbose:
                lines.append(
                    f"         rot {case.rotation:>6.1f} scale {case.scale:.2f} "
                    f"-> got {case.actual!r}"
                )
                if case.detail:
                    lines.append(f"            {case.detail}")

    failures = [result for result in results if not result.ok]
    lines.append("")
    lines.append(f"{len(results)} cases: {len(results) - len(failures)} passed, {len(failures)} failed")
    if failures:
        lines.append(
            "A failure means this hand shape is not recognised at that rotation or size. "
            "Re-run with --doctor, and consider tuning the [thresholds] section of the "
            "config: the HUD shows the live curl/reach values for each finger."
        )
    else:
        lines.append(
            "Every configured gesture is recognised across "
            f"{len(_SELF_TEST_ROTATIONS)} rotations and {len(_SELF_TEST_SCALES)} hand sizes."
        )
    return "\n".join(lines)
