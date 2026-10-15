"""Tests for the command-line interface.

The CLI is exercised through :func:`main` rather than by spawning a process, so
the tests are fast and cover the real argument handling. Any mode that would
start the interactive loop is avoided: those need a camera and a window.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import tomllib
import unittest
from pathlib import Path

from gesture_control.cli import build_parser, main


def run_cli(*argv: str) -> tuple[int, str, str]:
    """Run ``main`` capturing stdout and stderr."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(list(argv))
    return code, stdout.getvalue(), stderr.getvalue()


def parsed(*argv: str):
    """Parse arguments with argparse's usage messages suppressed."""
    with contextlib.redirect_stderr(io.StringIO()):
        return build_parser().parse_args(list(argv))


class TestParser(unittest.TestCase):
    def test_defaults_are_none_so_flags_do_not_clobber_the_config(self) -> None:
        """Every override must default to None.

        A default of ``0`` or ``False`` would silently overwrite a value that
        came from the config file, which is the classic layering bug.
        """
        args = parsed()
        for name in (
            "camera", "width", "height", "fps", "stride", "min_cutoff", "beta",
            "arm_seconds", "release_seconds", "repeat_delay", "repeat_interval",
            "volume_step", "brightness_step", "overlay_scale", "require_focus",
            "config", "preset", "source", "video", "detector", "max_frames",
        ):
            with self.subTest(option=name):
                self.assertIsNone(getattr(args, name), f"--{name} does not default to None")

    def test_boolean_flags_default_false(self) -> None:
        args = parsed()
        for name in ("doctor", "self_test", "print_config", "dry_run", "no_window", "no_mirror"):
            with self.subTest(option=name):
                self.assertFalse(getattr(args, name))

    def test_invalid_preset_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parsed("--preset", "nope")

    def test_invalid_choice_rejected(self) -> None:
        for argv in (["--source", "webcam"], ["--detector", "camera"], ["--log-level", "TRACE"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parsed(*argv)

    def test_help_exits_cleanly(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            parsed("--help")
        self.assertEqual(ctx.exception.code, 0)


class TestInformationModes(unittest.TestCase):
    def test_list_presets(self) -> None:
        code, out, _ = run_cli("--list-presets")
        self.assertEqual(code, 0)
        for preset in ("classic", "media", "minimal"):
            self.assertIn(preset, out)

    def test_list_gestures_shows_the_generated_table(self) -> None:
        """It must say what each gesture *does*, not just what it is called."""
        code, out, _ = run_cli("--list-gestures")
        self.assertEqual(code, 0)
        self.assertIn("make this shape", out)
        self.assertIn("and it does", out)
        self.assertIn("Play/Pause", out)  # an action label, not an id
        self.assertIn("TWO-HAND gestures", out)
        self.assertIn("Both hands, held", out)

    def test_list_gestures_shows_the_same_shape_differently_per_mode(self) -> None:
        """The listing has to make the mode concept legible from the terminal."""
        _, out, _ = run_cli("--list-gestures")
        self.assertIn("[1/5] MEDIA", out)
        self.assertIn("[2/5] SYSTEM", out)
        self.assertIn("Volume +", out)        # index in MEDIA
        self.assertIn("Brightness +", out)    # index in SYSTEM
        self.assertIn("Page down", out)       # index in BROWSE
        self.assertIn("Right arrow", out)     # index in PRESENT

    def test_list_actions_includes_backends(self) -> None:
        code, out, _ = run_cli("--list-actions")
        self.assertEqual(code, 0)
        self.assertIn("volume_up", out)
        self.assertIn("Backend capabilities:", out)

    def test_print_config_is_valid_toml(self) -> None:
        code, out, _ = run_cli("--print-config")
        self.assertEqual(code, 0)
        parsed = tomllib.loads(out)
        self.assertIn("camera", parsed)
        self.assertIn("modes", parsed)

    def test_print_config_reflects_flags(self) -> None:
        code, out, _ = run_cli("--print-config", "--volume-step", "0.02", "--no-mirror")
        self.assertEqual(code, 0)
        parsed = tomllib.loads(out)
        self.assertAlmostEqual(parsed["actions"]["volume_step"], 0.02)
        self.assertFalse(parsed["camera"]["mirror"])

    def test_version(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)

    def test_preset_flag_changes_the_gestures(self) -> None:
        _, classic, _ = run_cli("--list-gestures", "--preset", "classic")
        _, media, _ = run_cli("--list-gestures", "--preset", "media")
        self.assertIn("key_space", classic)  # only the classic preset has this
        self.assertIn("mute_toggle", media)
        self.assertNotIn("mute_toggle", classic)

    def test_default_preset_is_the_multi_mode_one(self) -> None:
        _, out, _ = run_cli("--list-gestures")
        self.assertIn("modes: 5", out)
        for mode in ("MEDIA", "SYSTEM", "BROWSE", "PRESENT", "WINDOW"):
            self.assertIn(mode, out)

    def test_list_modes_shows_the_switching_gestures(self) -> None:
        code, out, _ = run_cli("--list-modes")
        self.assertEqual(code, 0)
        self.assertIn("start mode: media", out)
        for label in ("BOTH OPEN", "BOTH FISTS", "BOTH PEACE", "BOTH SHAKA", "BOTH HORNS"):
            self.assertIn(label, out)

    def test_mode_flag_selects_the_start_mode(self) -> None:
        code, out, _ = run_cli("--list-modes", "--mode", "browse")
        self.assertEqual(code, 0)
        self.assertIn("start mode: browse", out)

    def test_unknown_mode_flag_is_reported(self) -> None:
        code, _, err = run_cli("--list-modes", "--mode", "nope")
        self.assertEqual(code, 1)
        self.assertIn("unknown mode", err)


class TestSelfTestMode(unittest.TestCase):
    def test_self_test_passes(self) -> None:
        code, out, _ = run_cli("--self-test")
        self.assertEqual(code, 0, out)
        self.assertIn("0 failed", out)

    def test_self_test_fails_on_bad_thresholds(self) -> None:
        """Exit code must reflect a genuine recognition failure."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            # Reach thresholds above what an extended finger achieves.
            path.write_text(
                "[thresholds]\nfinger_reach_up = 2.0\nfinger_reach_down = 1.85\n",
                encoding="utf-8",
            )
            code, out, _ = run_cli("--self-test", "--config", str(path))
        self.assertEqual(code, 1, out)
        self.assertIn("failed", out)


class TestConfigErrors(unittest.TestCase):
    def test_missing_config_file(self) -> None:
        code, _, err = run_cli("--config", "definitely-missing.toml", "--list-gestures")
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_bad_config_reports_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text("[camera]\nindex = 'front'\n", encoding="utf-8")
            code, _, err = run_cli("--config", str(path), "--list-gestures")
        self.assertEqual(code, 1)
        self.assertIn("expected an integer", err)

    def test_unknown_key_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text("[camera]\nnope = 1\n", encoding="utf-8")
            code, _, err = run_cli("--config", str(path), "--list-gestures")
        self.assertEqual(code, 1)
        self.assertIn("unknown option", err)

    def test_invalid_flag_value_is_reported(self) -> None:
        code, _, err = run_cli("--list-gestures", "--volume-step", "5")
        self.assertEqual(code, 1)
        self.assertIn("volume_step", err)


class TestDoctorMode(unittest.TestCase):
    def test_doctor_returns_a_status(self) -> None:
        """Exit code 0 when healthy, 1 when something failed; both are valid."""
        code, out, _ = run_cli("--doctor")
        self.assertIn(code, (0, 1))
        self.assertIn("checks:", out)
        self.assertIn("gesture-control", out)


class TestRunGuards(unittest.TestCase):
    def test_source_file_requires_a_video(self) -> None:
        code, _, err = run_cli("--source", "file", "--dry-run", "--no-window", "--max-frames", "1")
        self.assertEqual(code, 1)
        self.assertIn("--video", err)

    def test_missing_video_file_is_reported(self) -> None:
        code, _, err = run_cli(
            "--source", "file", "--video", "nope.mp4", "--dry-run", "--no-window",
            "--max-frames", "1",
        )
        self.assertEqual(code, 1)
        self.assertIn("not found", err)

    def test_headless_synthetic_run_completes(self) -> None:
        """The full loop, end to end, with no camera and no window.

        ``--dry-run`` is not optional here: without it the synthetic routine
        would synthesise real keystrokes into whatever window is focused.
        """
        code, out, err = run_cli(
            "--source", "test", "--detector", "synthetic", "--dry-run", "--no-window",
            "--max-frames", "90", "--volume-step", "0.05",
        )
        self.assertEqual(code, 0, err)
        self.assertIn("gesture-control", out)
        self.assertIn("frames in", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
