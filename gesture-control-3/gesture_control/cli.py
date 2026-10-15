"""Command-line interface.

Every knob in the config file has a matching flag, so a one-off experiment does
not require editing a file. Flags only override what they are given: an unset
flag leaves the config file's value alone, which is what makes layering
defaults -> preset -> file -> flags predictable.

Exit codes: ``0`` success, ``1`` a failure (including a failed self-test),
``2`` for a usage error, which :mod:`argparse` raises itself.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import MAX_PYTHON, MIN_PYTHON, __version__
from .config import (
    PRESETS,
    AppConfig,
    ConfigError,
    available_presets,
    config_to_toml,
    load_config,
)
from .diagnostics import format_doctor, format_self_test, run_doctor, run_self_test

__all__ = ["main", "build_parser", "setup_logging"]

_LOG = logging.getLogger("gesture_control")


def setup_logging(level: str, *, quiet: bool = False) -> None:
    """Configure root logging once, with a compact format."""
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)
    if quiet:
        root.setLevel(logging.ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-control",
        description=(
            "Control volume, brightness and media keys with hand gestures. "
            "Run with --doctor first to check what this machine supports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m gesture_control --doctor\n"
            "  python -m gesture_control --self-test\n"
            "  python -m gesture_control --preset media\n"
            "  python -m gesture_control --dry-run --no-window --max-frames 200\n"
            "  python -m gesture_control --source test --detector synthetic --no-window\n"
            "  python -m gesture_control --list-modes\n"
            "  python -m gesture_control --print-config > my-config.toml\n"
        ),
    )

    parser.add_argument("--version", action="version", version=f"gesture-control {__version__}")

    general = parser.add_argument_group("general")
    general.add_argument("--config", metavar="PATH", help="TOML configuration file")
    general.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="gesture preset to start from (default: classic)",
    )
    general.add_argument("--mode", metavar="NAME",
                         help="mode to start in (see --list-modes)")
    general.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="logging verbosity (default: INFO)")
    general.add_argument("--quiet", action="store_true", help="log errors only")
    general.add_argument("--debug", action="store_true",
                         help="shorthand for --log-level DEBUG with the detailed HUD")

    modes = parser.add_argument_group("modes (these exit immediately)")
    modes.add_argument("--doctor", action="store_true",
                       help="check the interpreter, dependencies, camera, and system backends")
    modes.add_argument("--self-test", action="store_true",
                       help="verify gesture recognition against the synthetic hand model")
    modes.add_argument("--print-config", action="store_true",
                       help="print the effective configuration as TOML")
    modes.add_argument("--list-presets", action="store_true", help="list gesture presets")
    modes.add_argument("--list-gestures", action="store_true",
                       help="list every mode and its gestures")
    modes.add_argument("--list-modes", action="store_true",
                       help="list modes and the two-hand gestures that switch them")
    modes.add_argument("--list-actions", action="store_true", help="list available actions")
    modes.add_argument("--verbose", action="store_true",
                       help="with --self-test, show every case rather than failures only")

    video = parser.add_argument_group("input")
    video.add_argument("--camera", type=int, metavar="N", help="camera index (default: 0)")
    video.add_argument("--source", choices=["camera", "test", "file"],
                       help="frame source (default: camera)")
    video.add_argument("--video", metavar="PATH", help="video file for --source file")
    video.add_argument("--width", type=int, help="requested capture width")
    video.add_argument("--height", type=int, help="requested capture height")
    video.add_argument("--fps", type=int, help="requested capture frame rate")
    video.add_argument("--no-mirror", action="store_true",
                       help="do not mirror the image (mirroring is on by default)")
    video.add_argument("--detector", choices=["auto", "mediapipe", "synthetic"],
                       help="hand detector backend (default: auto)")
    video.add_argument("--stride", type=int, metavar="N",
                       help="run detection every Nth frame (default: 1)")

    output = parser.add_argument_group("output")
    output.add_argument("--no-window", action="store_true",
                        help="run without the preview window")
    output.add_argument("--max-frames", type=int, metavar="N",
                        help="stop after N frames (useful for testing)")
    output.add_argument("--save-frames", metavar="DIR",
                        help="write rendered frames to DIR as PNG files")
    output.add_argument("--save-every", type=int, metavar="N", default=0,
                        help="with --save-frames, save every Nth frame (default: every frame)")
    output.add_argument("--overlay-scale", type=float, metavar="S",
                        help="scale the HUD text and panels (default: 1.0)")

    tuning = parser.add_argument_group("tuning")
    tuning.add_argument("--dry-run", action="store_true",
                        help="recognise gestures but never change anything")
    tuning.add_argument("--sync-actions", action="store_true",
                        help="run actions inline instead of on a worker thread")
    tuning.add_argument("--no-keyboard", action="store_true",
                        help="disable keyboard and media-key synthesis")
    tuning.add_argument("--require-focus", metavar="TITLE",
                        help="only send keys while a window title contains TITLE")
    tuning.add_argument("--no-smoothing", action="store_true", help="disable landmark smoothing")
    tuning.add_argument("--min-cutoff", type=float, metavar="HZ",
                        help="smoothing cutoff at rest; lower is smoother (default: 1.1)")
    tuning.add_argument("--beta", type=float, metavar="B",
                        help="smoothing velocity coupling (default: 0.035)")
    tuning.add_argument("--arm-seconds", type=float, metavar="S",
                        help="how long a gesture must be held before it fires (default: 0.10)")
    tuning.add_argument("--release-seconds", type=float, metavar="S",
                        help="how long it must be absent to count as released (default: 0.12)")
    tuning.add_argument("--repeat-delay", type=float, metavar="S",
                        help="delay before a held gesture starts repeating (default: 0.45)")
    tuning.add_argument("--repeat-interval", type=float, metavar="S",
                        help="spacing of repeats while a gesture is held (default: 0.12)")
    tuning.add_argument("--volume-step", type=float, metavar="F",
                        help="volume change per step, 0-1 (default: 0.05)")
    tuning.add_argument("--brightness-step", type=int, metavar="N",
                        help="brightness change per step, 1-100 (default: 8)")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply command-line overrides onto a loaded config.

    Only non-``None`` values are applied, so a flag that was not given cannot
    clobber a value that came from the config file.
    """
    if args.camera is not None:
        config.camera.index = args.camera
    if args.width is not None:
        config.camera.width = args.width
    if args.height is not None:
        config.camera.height = args.height
    if args.fps is not None:
        config.camera.fps = args.fps
    if args.no_mirror:
        config.camera.mirror = False

    if args.stride is not None:
        config.detector.stride = args.stride

    if args.no_smoothing:
        config.smoothing.enabled = False
    if args.min_cutoff is not None:
        config.smoothing.min_cutoff = args.min_cutoff
    if args.beta is not None:
        config.smoothing.beta = args.beta

    if args.arm_seconds is not None:
        config.tracking.arm_seconds = args.arm_seconds
    if args.release_seconds is not None:
        config.tracking.release_seconds = args.release_seconds
    if args.repeat_delay is not None:
        config.tracking.repeat_initial_delay = args.repeat_delay
    if args.repeat_interval is not None:
        config.tracking.repeat_interval = args.repeat_interval

    if args.dry_run:
        config.actions.dry_run = True
    if args.no_keyboard:
        config.actions.allow_keyboard = False
    if args.require_focus is not None:
        config.actions.require_focus_title = args.require_focus
    if args.volume_step is not None:
        config.actions.volume_step = args.volume_step
    if args.brightness_step is not None:
        config.actions.brightness_step = args.brightness_step

    if args.mode is not None:
        config.switching.start = args.mode

    if args.no_window:
        config.overlay.show_window = False
    if args.overlay_scale is not None:
        config.overlay.scale = args.overlay_scale
    if args.debug:
        config.overlay.show_timings = True
        config.overlay.show_finger_states = True

    config.validate()
    return config


def _list_gestures(config: AppConfig) -> str:
    """Print every mode with what each of its gestures does.

    The columns are shape name, what it does, and the action identifier. The
    middle column is the point: the listing should be readable without knowing
    the action naming scheme or decoding a finger diagram.
    """
    from .actions import build_registry

    registry, _ = build_registry(
        config.actions, mode_names=config.mode_names, async_actions=False
    )
    labels = registry.labels()

    lines = [f"preset: {config.preset}   modes: {len(config.modes)}", ""]
    for index, mode in enumerate(config.modes, start=1):
        marker = "  <- start" if config.switching.start.lower() == mode.name.lower() else ""
        lines.append(f"[{index}/{len(config.modes)}] {mode.label}  ({mode.name}){marker}")
        if mode.description:
            lines.append(f"    {mode.description}")
        lines.append("")
        lines.append(f"    {'make this shape':<18} {'and it does':<26} action")
        lines.append("    " + "-" * 76)
        for spec in mode:
            does = labels.get(spec.action, spec.action)
            lines.append(f"    {spec.label:<18} {does:<26} {spec.action}")
        lines.append("")

    lines.append(f"TWO-HAND gestures ({len(config.two_hand)}). Both hands, held: this is")
    lines.append("what makes mode changes and quitting impossible to trigger by accident.")
    lines.append("")
    lines.append(f"    {'make this shape':<18} {'and it does':<26} action")
    lines.append("    " + "-" * 76)
    for gesture in config.two_hand:
        does = labels.get(gesture.action, gesture.action)
        lines.append(f"    {gesture.label:<18} {does:<26} {gesture.action}")

    lines.append("")
    lines.append("On screen, the HUD draws a picture of each hand shape beside its")
    lines.append("action, and the table follows whichever mode you are in.")
    lines.append("Run --list-modes for just the modes, or --self-test to verify them.")
    return "\n".join(lines)


def _list_modes(config: AppConfig) -> str:
    lines = [
        f"preset: {config.preset}",
        f"start mode: {config.switching.start or config.mode_names[0]}"
        f"   wrap: {config.switching.wrap}",
        "",
        f"{'#':<3} {'name':<12} {'label':<12} {'gestures':<9} description",
        "-" * 88,
    ]
    for index, mode in enumerate(config.modes, start=1):
        lines.append(
            f"{index:<3} {mode.name:<12} {mode.label:<12} {len(mode):<9} {mode.description}"
        )
    lines.append("")
    lines.append("Switch modes with a two-hand gesture:")
    for gesture in config.two_hand:
        lines.append(f"  {gesture.label:<12} {gesture.description}")
    return "\n".join(lines)


def _list_actions(config: AppConfig) -> str:
    from .actions import build_registry

    registry, capabilities = build_registry(
        config.actions, mode_names=config.mode_names, async_actions=False
    )
    lines = ["Available actions:", ""]
    lines.extend(f"  {line}" for line in registry.describe_all())
    lines.append("")
    lines.append("Backend capabilities:")
    for subsystem, entries in capabilities.items():
        for capability in entries:
            lines.append(f"  {subsystem:<11} {capability.describe()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, quiet=args.quiet)

    # -- information modes ------------------------------------------------- #
    if args.list_presets:
        print("Gesture presets:")
        for name, description in available_presets().items():
            print(f"  {name:<10} {description}")
        print("\nUse --preset NAME, or set 'preset = \"NAME\"' in a config file.")
        return 0

    try:
        config = load_config(args.config, preset=args.preset)
        config = _apply_overrides(config, args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1

    if args.print_config:
        print(config_to_toml(config))
        return 0
    if args.list_gestures:
        print(_list_gestures(config))
        return 0
    if args.list_modes:
        print(_list_modes(config))
        return 0
    if args.list_actions:
        print(_list_actions(config))
        return 0

    _warn_about_interpreter()

    if args.doctor:
        checks = run_doctor(config)
        print(format_doctor(checks))
        return 1 if any(check.status == "fail" for check in checks) else 0
    if args.self_test:
        results = run_self_test(config)
        print(format_self_test(results, verbose=args.verbose))
        return 1 if any(not result.ok for result in results) else 0

    return _run(config, args)


def _warn_about_interpreter() -> None:
    """Warn early if this interpreter is outside the tested range.

    On a machine with several Python installations, a version mismatch is by far
    the most common failure, and the traceback that eventually appears names a
    module rather than the interpreter, which sends people looking in the wrong
    place.

    The wording is careful to say *untested* rather than *broken*. Whether a
    given interpreter works is a fact that can be checked rather than assumed,
    and by the time this runs MediaPipe has already imported successfully.
    """
    version = sys.version_info[:2]
    if MIN_PYTHON <= version <= MAX_PYTHON:
        return

    mediapipe_status = ""
    try:
        import mediapipe  # noqa: F401

        mediapipe_status = " MediaPipe imports here, so it will be used."
    except Exception:
        mediapipe_status = " MediaPipe is NOT importable here, so this will fail."

    print(
        f"warning: Python {version[0]}.{version[1]} is outside the range this project is "
        f"tested against\n"
        f"         ({MIN_PYTHON[0]}.{MIN_PYTHON[1]} - {MAX_PYTHON[0]}.{MAX_PYTHON[1]}, the "
        f"versions MediaPipe publishes wheels for).{mediapipe_status}\n"
        f"         interpreter: {sys.executable}\n"
        f"         Run run.py to use a supported interpreter instead.",
        file=sys.stderr,
    )


def _run(config: AppConfig, args: argparse.Namespace) -> int:
    """Build and run the controller. Returns an exit code."""
    from .actions.base import ActionError
    from .app import GestureApp
    from .detector import DetectorError
    from .sources import SourceError

    source_kind = args.source or "camera"
    if source_kind == "file" and not args.video:
        print("error: --source file requires --video PATH", file=sys.stderr)
        return 1

    try:
        app = GestureApp(
            config,
            source_kind=source_kind,
            source_path=args.video,
            detector_kind=args.detector,
        )
    except (ConfigError, DetectorError, SourceError, ActionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total_gestures = sum(len(mode) for mode in app.modes)
    print(
        f"gesture-control {__version__} - preset {config.preset!r}, "
        f"{len(app.modes)} mode(s), {total_gestures} single-hand gesture(s), "
        f"{len(app.two_hand)} two-hand gesture(s)"
    )
    print(f"start mode: {app.mode_label}   switch: {', '.join(g.label for g in app.two_hand)}")
    for warning in (*app.mode_set.warnings, *app.two_hand_set.warnings):
        _LOG.warning("%s", warning)
    for action_id, subsystem in (("volume_up", "volume"), ("brightness_up", "brightness")):
        backend = getattr(app.registry.get(action_id), "backend", None)
        if backend is not None and not backend.available:
            print(
                f"note: {subsystem} control is unavailable ({backend.capability.reason}).\n"
                f"      Run --doctor for details.",
                file=sys.stderr,
            )
    if config.actions.dry_run:
        print("dry run: gestures are recognised but nothing will be changed")

    print(
        "keys: q/Esc quit, p or space pause, n/b next/previous mode, h first mode, "
        "d details, m dry run, s screenshot"
    )
    try:
        summary = app.run(
            max_frames=args.max_frames,
            save_frames=args.save_frames,
            save_every=args.save_every or 0,
        )
    except KeyboardInterrupt:
        print("\ninterrupted")
        app.shutdown()
        return 0
    except DetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        app.shutdown()
        return 1
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        app.shutdown()
        return 1

    print(summary.describe())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
