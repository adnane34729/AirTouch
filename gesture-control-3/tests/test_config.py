"""Tests for configuration loading, validation, presets and TOML round-tripping."""

from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from gesture_control.config import (
    PRESETS,
    AppConfig,
    ConfigError,
    available_presets,
    config_to_toml,
    default_config,
    iter_config_paths,
    load_config,
)


class TestDefaults(unittest.TestCase):
    def test_every_preset_loads(self) -> None:
        for preset in PRESETS:
            with self.subTest(preset=preset):
                config = default_config(preset)
                self.assertEqual(config.preset, preset)
                self.assertTrue(config.modes, "a preset must define modes")

    def test_unknown_preset_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            default_config("nope")

    def test_available_presets_have_descriptions(self) -> None:
        presets = available_presets()
        self.assertEqual(set(presets), set(PRESETS))
        for description in presets.values():
            self.assertTrue(description)

    def test_classic_matches_the_original_defaults(self) -> None:
        """Documented departures from the original, asserted so they are visible."""
        config = default_config("classic")
        # Original: cv2.VideoCapture(0), min_detection_confidence=0.5
        self.assertEqual(config.camera.index, 0)
        self.assertEqual(config.detector.min_detection_confidence, 0.5)
        # Two, not the original's single hand: the two-hand pause and quit
        # gestures cannot work if a second hand is never detected.
        self.assertEqual(config.detector.max_hands, 2)
        # Original: action_delay = 1.0 s globally. Now per-gesture and repeatable.
        self.assertLess(config.tracking.default_cooldown, 1.0)
        self.assertGreater(config.tracking.repeat_interval, 0.0)

    def test_all_option_paths_are_enumerable(self) -> None:
        paths = set(iter_config_paths(default_config()))
        for expected in (
            "camera.index",
            "detector.max_hands",
            "smoothing.beta",
            "tracking.arm_seconds",
            "thresholds.finger_curl_up",
            "actions.volume_step",
            "overlay.show_skeleton",
        ):
            self.assertIn(expected, paths)


class TestTomlRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_every_value(self) -> None:
        for preset in PRESETS:
            with self.subTest(preset=preset):
                original = default_config(preset)
                text = config_to_toml(original)
                parsed = tomllib.loads(text)  # must be valid TOML
                self.assertEqual(parsed["preset"], preset)

                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "config.toml"
                    path.write_text(text, encoding="utf-8")
                    loaded = load_config(path, preset=preset)
                self.assertEqual(loaded, original)

    def test_printed_config_is_documented(self) -> None:
        text = config_to_toml(default_config("classic"))
        self.assertIn("requirements", text)
        self.assertIn("[[modes]]", text)
        self.assertIn("#", text)

    def test_gestures_survive_the_round_trip(self) -> None:
        original = default_config("media")
        text = config_to_toml(original)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text(text, encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(
            [(m.name, [(s.name, s.action, s.pose_text()) for s in m]) for m in loaded.modes],
            [(m.name, [(s.name, s.action, s.pose_text()) for s in m]) for m in original.modes],
        )


class TestLoading(unittest.TestCase):
    def _load(self, text: str, **kwargs) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path, **kwargs)

    def test_replacing_modes_clears_a_dangling_start_mode(self) -> None:
        """A custom mode list must not inherit the preset's start mode name."""
        config = self._load(
            """
            [[modes]]
            name = "solo"
            label = "SOLO"

            [[modes.gestures]]
            name = "g"
            action = "volume_up"
            requirements = "index:up"
            """
        )
        self.assertEqual(config.switching.start, "")
        self.assertEqual(config.mode_names, ("solo",))
        config.validate()

    def test_explicit_start_mode_survives_custom_modes(self) -> None:
        config = self._load(
            """
            [switching]
            start = "second"

            [[modes]]
            name = "first"

            [[modes.gestures]]
            name = "g"
            action = "volume_up"
            requirements = "index:up"

            [[modes]]
            name = "second"

            [[modes.gestures]]
            name = "g"
            action = "volume_down"
            requirements = "pinky:up"
            """
        )
        self.assertEqual(config.switching.start, "second")

    def test_unknown_start_mode_is_reported(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self._load('[switching]\nstart = "nope"\n')
        self.assertIn("unknown mode", str(ctx.exception))

    def test_overrides_a_single_value(self) -> None:
        config = self._load("[camera]\nindex = 3\n")
        self.assertEqual(config.camera.index, 3)
        self.assertEqual(config.camera.width, 1280)  # untouched

    def test_unknown_key_is_an_error_with_guidance(self) -> None:
        """A silently ignored typo is how the original lost hours; not here."""
        with self.assertRaises(ConfigError) as ctx:
            self._load("[camera]\nidx = 3\n")
        message = str(ctx.exception)
        self.assertIn("unknown option 'idx'", message)
        self.assertIn("index", message)

    def test_wrong_type_is_an_error(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self._load('[camera]\nindex = "front"\n')
        self.assertIn("expected an integer", str(ctx.exception))

    def test_bool_is_not_accepted_as_int(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[camera]\nwidth = true\n")

    def test_int_is_accepted_for_float(self) -> None:
        config = self._load("[tracking]\narm_seconds = 1\n")
        self.assertEqual(config.tracking.arm_seconds, 1.0)

    def test_malformed_toml_is_reported(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self._load("[camera\nindex = 0\n")
        self.assertIn("not valid TOML", str(ctx.exception))

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_config(Path("does-not-exist.toml"))
        self.assertIn("not found", str(ctx.exception))

    def test_preset_from_file(self) -> None:
        config = self._load('preset = "media"\n')
        self.assertEqual(config.preset, "media")
        self.assertIn("mute", [s.name for m in config.modes for s in m])

    def test_explicit_preset_overrides_the_file(self) -> None:
        """An explicit command-line choice must not be reversed by the file."""
        config = self._load('preset = "media"\n', preset="classic")
        self.assertEqual(config.preset, "classic")
        # 'next_track' exists only in the media preset.
        self.assertNotIn("next_track", [s.name for m in config.modes for s in m])
        self.assertIn("space", [s.name for m in config.modes for s in m])

    def test_unknown_preset_in_file(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self._load('preset = "bogus"\n')
        self.assertIn("unknown preset", str(ctx.exception))

    def test_custom_gestures_replace_the_preset(self) -> None:
        config = self._load(
            """
            [[gestures]]
            name = "blink"
            action = "mute_toggle"
            requirements = "index:up,middle:up,ring:up,pinky:up"
            label = "ALL"
            """
        )
        self.assertEqual([s.name for m in config.modes for s in m], ["blink"])

    def test_empty_gesture_list_is_rejected(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self._load("gestures = []\n")
        self.assertIn("non-empty", str(ctx.exception))

    def test_gesture_requirements_accepts_a_table(self) -> None:
        config = self._load(
            """
            [[gestures]]
            name = "t"
            action = "volume_up"

            [gestures.requirements]
            index = "up"
            middle = "down"
            """
        )
        from gesture_control.types import Finger

        self.assertEqual(config.modes[0].gestures[0].requirements[Finger.INDEX].value, "up")

    def test_bad_gesture_requirement_is_reported_with_context(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self._load(
                """
                [[gestures]]
                name = "t"
                action = "volume_up"
                requirements = "elbow:up"
                """
            )
        self.assertIn("t", str(ctx.exception))

    def test_comments_are_ignored(self) -> None:
        config = self._load("# a comment\n[camera]\nindex = 2 # inline\n")
        self.assertEqual(config.camera.index, 2)


class TestValidation(unittest.TestCase):
    def _load(self, text: str) -> AppConfig:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_inverted_finger_thresholds_rejected(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            self._load("[thresholds]\nfinger_curl_up = 100\nfinger_curl_down = 150\n")
        self.assertIn("finger_curl_down", str(ctx.exception))

    def test_inverted_reach_thresholds_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[thresholds]\nfinger_reach_up = 0.5\nfinger_reach_down = 0.9\n")

    def test_inverted_thumb_curl_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[thresholds]\nthumb_curl_up = 100\nthumb_curl_down = 150\n")

    def test_inverted_thumb_abduction_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[thresholds]\nthumb_abduction_up = 0.1\nthumb_abduction_down = 0.9\n")

    def test_zero_stride_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[detector]\nstride = 0\n")

    def test_bad_volume_step_rejected(self) -> None:
        for value in ("0.0", "1.5", "-0.2"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self._load(f"[actions]\nvolume_step = {value}\n")

    def test_bad_brightness_step_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[actions]\nbrightness_step = 0\n")

    def test_bad_confidence_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[detector]\nmin_detection_confidence = 1.5\n")

    def test_bad_overlay_scale_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[overlay]\nscale = 0.0\n")

    def test_bad_model_complexity_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._load("[detector]\nmodel_complexity = 2\n")

    def test_gesture_conflict_is_reported_at_load(self) -> None:
        with self.assertRaises(ConfigError):
            self._load(
                """
                [[gestures]]
                name = "a"
                action = "volume_up"
                requirements = "index:up"

                [[gestures]]
                name = "b"
                action = "volume_down"
                requirements = "index:up,pinky:any"
                """
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
