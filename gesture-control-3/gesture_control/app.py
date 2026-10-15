"""The application: capture, detect, interpret, act, display.

Robustness differences from the original loop:

* **Guaranteed cleanup.** ``try/finally`` releases the camera and closes the
  detector and the window on *any* exit path, including an exception. The
  original called ``cap.release()`` only after a clean loop exit, so a crash
  left the camera locked until the process died.
* **Camera recovery.** A failed ``read()`` triggers a reopen with backoff instead
  of ``break``.
* **Human-readable startup failures.** No camera, no MediaPipe and no working
  volume backend each produce an explanation rather than silence.
* **Hotkeys.** ``space`` pauses (resetting temporal state so nothing fires on
  resume), ``d`` toggles the detail readout, ``m`` toggles dry-run, ``s`` saves
  a screenshot, ``q``/``Esc`` quit.
* **Nothing blocks.** Actions run off-thread, so a slow WMI call cannot stutter
  the preview.
* **Frame skipping.** ``detector.stride`` runs detection every Nth frame while
  still displaying every frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .actions import ActionRegistry, build_registry, resolve_modes, resolve_timing
from .config import AppConfig
from .detector import DetectorError, HandDetector, create_detector
from .geometry import make_observation
from .gestures import GestureSet, GestureSpec
from .modes import Mode, ModeError, ModeSet, TwoHandGesture, TwoHandGestureSet
from .overlay import HudRenderer, HudState
from .pose import PoseEstimator
from .smoothing import MultiHandSmoother
from .sources import FrameSource, create_source
from .tracking import GestureDebouncer, Trigger
from .types import Handedness, HandObservation, HandPose

__all__ = ["GestureApp", "RunSummary", "ReadoutCache"]

_LOG = logging.getLogger(__name__)

#: ``p`` is what people reach for; ``space`` was the original binding. Both work.
_KEY_QUIT = (ord("q"), 27)  # q, Esc
_KEY_PAUSE = (ord("p"), ord(" "))
_KEY_MODE_NEXT = (ord("n"), ord("."), ord("]"))
_KEY_MODE_PREV = (ord("b"), ord(","), ord("["))
_KEY_MODE_HOME = (ord("h"), ord("/"))
_KEY_DEBUG = ord("d")
_KEY_DRYRUN = ord("m")
_KEY_SCREENSHOT = ord("s")
_KEY_RESET = ord("r")

#: Actions that only make sense if the HUD shows a level, and the subsystem each
#: one belongs to.
_LEVEL_ACTIONS = (("volume_up", "volume"), ("brightness_up", "brightness"))


def _cv2():
    """Import OpenCV lazily so the logic modules never require it."""
    import cv2

    return cv2


class ReadoutCache:
    """Reads system levels at most every ``interval`` seconds.

    Querying volume means a COM round trip and querying brightness can mean a WMI
    round trip of several hundred milliseconds. Doing that every frame -- as an
    unguarded HUD would -- is a real performance bug, so reads are cached and
    refreshed on a timer. The cache is also updated from action outcomes, so the
    display reacts immediately to a change the user just caused.
    """

    def __init__(self, interval: float = 1.5) -> None:
        self.interval = interval
        self.volume: float | None = None
        self.brightness: float | None = None
        self._last_read = 0.0
        self.reads = 0

    # -- reading ----------------------------------------------------------- #

    @staticmethod
    def _backend_percent(registry: ActionRegistry, action_id: str) -> float | None:
        action = registry.get(action_id)
        backend = getattr(action, "backend", None)
        if backend is None or not getattr(backend, "available", False):
            return None
        try:
            return backend.get_percent()
        except Exception as exc:
            _LOG.debug("could not read %s level: %s", action_id, exc)
            return None

    def prime(self, registry: ActionRegistry) -> None:
        """Populate once at startup so the first HUD frame is not empty."""
        self.volume = self._backend_percent(registry, "volume_up")
        self.brightness = self._backend_percent(registry, "brightness_up")
        self._last_read = time.monotonic()

    def maybe_refresh(self, registry: ActionRegistry, now: float) -> None:
        if now - self._last_read < self.interval:
            return
        self._last_read = now
        self.reads += 1
        self.volume = self._backend_percent(registry, "volume_up")
        self.brightness = self._backend_percent(registry, "brightness_up")

    def note_outcome(self, action_id: str, value: float | None) -> None:
        """Apply a value reported by an action, avoiding a stale HUD."""
        if value is None:
            return
        if action_id.startswith("volume"):
            self.volume = value
        elif action_id.startswith("brightness"):
            self.brightness = value


@dataclass
class RunSummary:
    """What happened during a run, for the closing log line and the tests."""

    frames: int = 0
    hands_seen: int = 0
    hands_lost: int = 0
    triggers: int = 0
    repeats: int = 0
    actions_ok: int = 0
    actions_failed: int = 0
    duration: float = 0.0
    mean_fps: float = 0.0
    reconnects: int = 0
    saved_frames: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = [
            f"{self.frames} frames in {self.duration:.1f}s ({self.mean_fps:.1f} fps)",
            f"{self.hands_seen} with a hand",
            f"{self.triggers} gesture(s) fired ({self.repeats} repeats)",
            f"{self.actions_ok} action(s) ok",
        ]
        if self.actions_failed:
            parts.append(f"{self.actions_failed} failed")
        if self.reconnects:
            parts.append(f"{self.reconnects} reconnect(s)")
        return ", ".join(parts)


class GestureApp:
    """Wires the pipeline together and runs the frame loop."""

    def __init__(
        self,
        config: AppConfig,
        *,
        source: FrameSource | None = None,
        detector: HandDetector | None = None,
        source_kind: str = "camera",
        source_path: str | Path | None = None,
        detector_kind: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.config.validate()
        # Injectable so tests can drive ``run`` with a synthetic timeline. The
        # scripted hand routine is time-based, so without this a headless run
        # would be paced by the CPU rather than by the script.
        self._clock = clock

        # Mode state must exist before the registry is built: the mode actions
        # report ``mode_count`` and ``mode_label`` in their own descriptions.
        self._mode_index = config.switching.start_index(config.mode_names)
        self.registry, self.capabilities = build_registry(
            config.actions, controller=self, mode_names=config.mode_names, async_actions=True
        )
        self._resolve_gestures()
        # A generated test source pairs with the synthetic detector, since there
        # is no real hand in a test pattern to find.
        kind = detector_kind or (
            "synthetic" if source_kind in ("test", "synthetic") else "auto"
        )
        self.detector = detector or create_detector(kind, config.detector)
        self.source = source or create_source(
            source_kind,
            camera=config.camera,
            path=source_path,
            width=config.camera.width or 1280,
            height=config.camera.height or 720,
        )

        self.estimator = PoseEstimator(config.build_thresholds())
        # One smoother per hand. A single shared smoother would blend both hands'
        # landmarks into nonsense, so two-hand support requires this.
        self.smoother = MultiHandSmoother(
            enabled=config.smoothing.enabled,
            min_cutoff=config.smoothing.min_cutoff,
            beta=config.smoothing.beta,
            d_cutoff=config.smoothing.d_cutoff,
        )
        self.debouncer = GestureDebouncer(
            arm_seconds=config.tracking.arm_seconds,
            release_seconds=config.tracking.release_seconds,
            repeat_initial_delay=config.tracking.repeat_initial_delay,
            repeat_interval=config.tracking.repeat_interval,
            default_cooldown=config.tracking.default_cooldown,
        )
        # Two-hand gestures get their own, slower debouncer: mode changes and
        # quitting should require a pose you deliberately held, not one that
        # flashed past while you moved your hands.
        self.two_hand_debouncer = GestureDebouncer(
            arm_seconds=config.tracking.two_hand_arm_seconds,
            release_seconds=config.tracking.two_hand_release_seconds,
            repeat_initial_delay=config.tracking.repeat_initial_delay,
            repeat_interval=config.tracking.repeat_interval,
            default_cooldown=config.tracking.default_cooldown,
        )
        self.hud = HudRenderer(config.overlay)
        self.readouts = ReadoutCache()
        self.summary = RunSummary()
        # The legend names what each gesture *does*, so the HUD never asks the
        # user to decode an action identifier.
        self.action_labels = self.registry.labels()

        self.paused = False
        self.registry.dry_run = config.actions.dry_run

        self._poses: dict[str, HandPose] = {}
        self._fps = 0.0
        self._frame_times: list[float] = []
        self._frame_shape: tuple[int, int] | None = None
        self._last_display: np.ndarray | None = None
        self._banner = ""
        self._banner_age = 99.0
        self._banner_ok = True
        self._action_id = ""
        self._action_message = ""
        self._action_ok = True
        self._action_time = 0.0
        self._mode_announce = ""
        self._mode_announce_time = 0.0
        self._mode_change_time = -1e9
        self._mode_switches = 0
        self._stopping = False
        self._opened = False
        self._save_dir: Path | None = None
        self._saved_count = 0
        self._warned: set[str] = set()

    # -- construction ------------------------------------------------------ #

    def _resolve_gestures(self) -> None:
        """Link every gesture -- in every mode, plus two-hand -- to real actions.

        Validation happens here rather than lazily, so a config that names a
        misspelled action fails at startup with the mode and gesture named.
        """
        self.modes: tuple[Mode, ...] = resolve_modes(self.config.modes, self.registry)
        self.two_hand: tuple[TwoHandGesture, ...] = resolve_timing(
            self.config.two_hand, self.registry, context="two-hand gesture"
        )
        self.config.modes = self.modes
        self.config.two_hand = self.two_hand
        self.mode_set = ModeSet(modes=self.modes, wrap=self.config.switching.wrap)
        self.two_hand_set = TwoHandGestureSet(gestures=self.two_hand)
        self.mode_gesture_sets: dict[str, GestureSet | None] = {
            mode.name: mode.build_gesture_set() for mode in self.modes
        }

    # -- controller interface (used by the application-control actions) ----- #

    @property
    def mode_count(self) -> int:
        return len(self.modes)

    @property
    def current_mode(self) -> Mode:
        return self.modes[self._mode_index]

    @property
    def current_mode_index(self) -> int:
        return self._mode_index

    @property
    def mode_label(self) -> str:
        return self.current_mode.label

    def set_paused(self, paused: bool) -> None:
        """Enter or leave the paused state.

        Note what this does *not* do: reset the debouncers. Doing so was a real
        bug. Resetting cleared the action's cooldown history, so the pause gesture
        -- still being held -- re-armed immediately and toggled the state back,
        flipping the application between paused and running while the user held
        the pose. Letting the hold stand means a one-shot gesture fires once per
        hold, which is what "hold this shape to pause" should mean.

        The queued-action drain is the part that matters: it stops a system action
        already in flight from landing after the pause took effect.
        """
        paused = bool(paused)
        if paused == self.paused:
            return
        self.paused = paused
        if paused:
            self.registry.drain(0.5)

    def step_mode(self, delta: int) -> str:
        """Move ``delta`` modes and announce the change."""
        if self.mode_count == 0:  # pragma: no cover - validation forbids it
            return "-"
        self._set_mode_index(self.mode_set.step(self._mode_index, delta))
        return self.mode_label

    def goto_mode(self, name: str) -> str:
        """Switch to a named mode."""
        try:
            index = self.mode_set.index_of(name)
        except ModeError as exc:
            raise ConfigError(str(exc)) from exc
        self._set_mode_index(index)
        return self.mode_label

    def home_mode(self) -> str:
        """Switch back to the first mode."""
        self._set_mode_index(0)
        return self.mode_label

    def request_quit(self) -> None:
        """Stop the frame loop at the end of the current iteration."""
        self._stopping = True

    def _set_mode_index(self, index: int) -> None:
        if index == self._mode_index:
            return
        self._mode_index = index
        self._mode_switches += 1
        self._mode_announce = self.current_mode.label
        self._mode_announce_time = self._clock()
        self._mode_change_time = self._mode_announce_time
        _LOG.info("mode: %s (%d/%d)", self.mode_label, index + 1, self.mode_count)

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> None:
        """Open the frame source and prime the readouts."""
        if self._opened:
            return
        self.source.open()
        self.readouts.prime(self.registry)
        self._opened = True

    def shutdown(self) -> None:
        """Release everything. Idempotent, and safe to call after a failure."""
        self._stopping = True
        for label, closer in (
            ("source", self.source.close),
            ("detector", self.detector.close),
            ("registry", self.registry.close),
        ):
            try:
                closer()
            except Exception:
                _LOG.debug("%s close failed", label, exc_info=True)
        if self.config.overlay.show_window:
            try:
                _cv2().destroyAllWindows()
            except Exception:
                pass

    close = shutdown

    # -- per-frame processing ---------------------------------------------- #

    def process(self, frame: np.ndarray, now: float, *, detect: bool = True) -> list[Trigger]:
        """Run one frame through the whole pipeline.

        Order matters here:

        1. **Two-hand gestures first.** If both hands are making a two-hand
           shape, that wins and no single-hand action fires. This is what makes
           mode switching and quitting unambiguous, and it also stops a hand
           passing through a single-hand shape on its way into a two-hand one.
        2. Otherwise, single-hand gestures from the *current mode*.
        3. While paused, only gestures whose action declares
           ``works_while_paused`` are considered at all.

        Separate from :meth:`run` so the integration tests can drive it directly
        with synthetic frames and a controlled clock.
        """
        if detect:
            self._update_poses(self._detect(frame, now), now)

        two_hand_candidate: TwoHandGesture | None = None
        single_candidate: GestureSpec | None = None

        pair = self._two_hand_pair()
        if pair is not None:
            candidate = self.two_hand_set.classify(pair[0], pair[1])
            if candidate is not None:
                two_hand_candidate = candidate

        if two_hand_candidate is None:
            single_candidate = self._classify_single_hand(pair is not None, now)

        # The pause filter is applied to *both* kinds, and only here. Filtering
        # only the two-hand candidate -- which is what an earlier version did --
        # left single-hand gestures firing while paused, which is exactly the
        # state the pause is supposed to stop.
        if two_hand_candidate is not None and not self._may_fire(two_hand_candidate.action):
            two_hand_candidate = None
        if single_candidate is not None and not self._may_fire(single_candidate.action):
            single_candidate = None

        triggers = self.two_hand_debouncer.update(
            two_hand_candidate, now, paused=False
        )
        triggers += self.debouncer.update(single_candidate, now, paused=False)
        for trigger in triggers:
            self._dispatch(trigger, now)
        return triggers

    def _may_fire(self, action_id: str) -> bool:
        """Whether an action is permitted in the current paused state."""
        if not self.paused:
            return True
        action = self.registry.get(action_id)
        if action is None or not action.works_while_paused:
            return False
        if action_id.startswith("mode_") and not self.config.switching.allow_while_paused:
            return False
        return True

    def _two_hand_pair(self) -> tuple[HandPose, HandPose] | None:
        """The two current poses, when two hands are visible."""
        if len(self._poses) < 2:
            return None
        poses = list(self._poses.values())[:2]
        return poses[0], poses[1]

    def _classify_single_hand(
        self, two_hands_visible: bool, now: float
    ) -> GestureSpec | None:
        """Classify single-hand gestures for the current mode.

        With two hands visible the configured policy decides whose hand counts,
        because allowing both would mean two actions firing from one pose and
        would make a two-hand gesture's transition frames destructive.

        Immediately after a mode change nothing is classified at all: the pose
        that triggered the change would otherwise be re-read against the new
        mode's table and fire a second action.
        """
        if not self._poses:
            return None
        if now - self._mode_change_time < self.config.switching.settle_seconds:
            return None

        policy = self.config.switching.single_hand_with_two_visible
        if two_hands_visible and policy == "suppress":
            return None

        if two_hands_visible and policy == "primary":
            pose = self._primary_pose()
        elif two_hands_visible and policy == "both":
            # First hand that matches anything, in a stable order.
            for pose in self._poses.values():
                match = self._classify_in_mode(pose)
                if match is not None:
                    return match
            return None
        else:
            pose = self._primary_pose()

        return None if pose is None else self._classify_in_mode(pose)

    def _classify_in_mode(self, pose: HandPose | None) -> GestureSpec | None:
        gesture_set = self.mode_gesture_sets.get(self.current_mode.name)
        return None if gesture_set is None else gesture_set.classify(pose)

    def _primary_pose(self) -> HandPose | None:
        """The pose used for single-hand gestures.

        Prefers the configured primary hand, then the left hand, then whatever is
        visible, so a single hand of either kind always works.
        """
        if not self._poses:
            return None
        if len(self._poses) == 1:
            return next(iter(self._poses.values()))

        wanted = self.config.switching.primary_hand.strip().lower()
        if wanted in ("left", "right"):
            key = wanted.capitalize()
            if key in self._poses:
                return self._poses[key]
        for fallback in ("Right", "Left"):
            if fallback in self._poses:
                return self._poses[fallback]
        return next(iter(self._poses.values()))

    def _detect(self, frame: np.ndarray, now: float) -> list[HandObservation]:
        try:
            return self.detector.detect(frame, now)
        except DetectorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._warn_once(f"detector error: {exc}")
            return []

    @staticmethod
    def _hand_key(observation: HandObservation, index: int) -> str:
        """Stable per-hand key used for smoothing history and pose slots.

        Handedness when the detector is confident, otherwise a position-derived
        slot. Either way the key must not change frame to frame, or the smoother
        would restart its filter every frame.
        """
        if observation.handedness is not Handedness.UNKNOWN:
            return observation.handedness.value
        wrist_x = float(observation.raw[0, 0]) if observation.raw.size else 0.5
        return "unknown-left" if wrist_x < 0.5 else "unknown-right"

    def _update_poses(self, observations: list[HandObservation], now: float) -> None:
        """Smooth and estimate every visible hand, tracking hands coming and going.

        Resetting a hand's smoother and hysteresis memory when it leaves is a
        small detail with a large effect: without it the first frames after a hand
        reappears are dragged toward its last known position, which can make a
        gesture fire spuriously on re-entry.

        The **debouncers are deliberately not reset** here. Detectors drop frames
        constantly, and clearing them on every dropout would restart the arming
        timer and discard a held gesture, making the controls unreliable in
        exactly the conditions they need to be robust in. They have
        ``release_seconds`` for this: a dropped frame simply arrives as ``None``.
        """
        if not observations:
            if self._poses:
                self.summary.hands_lost += 1
                self._reset_pose_state()
            return

        limit = max(1, self.config.detector.max_hands)
        seen: dict[str, HandPose] = {}
        for index, observation in enumerate(observations[:limit]):
            key = self._hand_key(observation, index)
            if key in seen:
                # Two detections claiming the same hand: keep the more confident.
                if observation.handedness_score <= seen[key].observation.handedness_score:
                    continue
            self.summary.hands_seen += 1
            smoothed = self.smoother.smooth(observation.raw, now, key=key)
            if smoothed is not observation.raw:
                observation = make_observation(
                    smoothed,
                    aspect_ratio=observation.aspect_ratio,
                    handedness=observation.handedness,
                    handedness_score=observation.handedness_score,
                    timestamp=observation.timestamp,
                )
            seen[key] = self.estimator.estimate(observation)

        for key in self._poses.keys() - seen.keys():
            self.smoother.forget(key)
            self.estimator.reset(self._poses[key].handedness)
        self._poses = seen

    def _dispatch(self, trigger: Trigger, now: float) -> None:
        """Perform a trigger's action and record it for the HUD."""
        spec = trigger.gesture
        self.summary.triggers += 1
        if trigger.is_repeat:
            self.summary.repeats += 1

        outcome = self.registry.invoke(spec.action)

        # An asynchronously queued action has no result yet. Show the last known
        # result for that action rather than the word "queued", which would tell
        # the user nothing about whether it worked.
        if outcome.message == "queued":
            display = self.registry.last_outcome.get(spec.action, outcome)
            display_ok = display.ok
        else:
            display = outcome
            display_ok = outcome.ok

        self._action_id = spec.label or spec.name
        self._action_message = display.message
        self._action_ok = display_ok
        self._action_time = now

        if display_ok:
            self.readouts.note_outcome(spec.action, display.value)

        suffix = " (repeat)" if trigger.is_repeat else ""
        self._banner = f"{spec.label or spec.name}: {display.message}{suffix}"
        self._banner_age = 0.0
        self._banner_ok = display_ok
        _LOG.info(
            "gesture %s -> %s: %s%s",
            spec.name, spec.action, display.message, " [repeat]" if trigger.is_repeat else "",
        )

    def _warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            _LOG.warning("%s", message)

    def _reset_pose_state(self) -> None:
        """Forget per-hand state. Called when every hand leaves the frame."""
        self.smoother.reset()
        self.estimator.reset()
        self._poses = {}

    def _reset_pipeline(self) -> None:
        """Forget *all* temporal state, including held gestures.

        Used for pause/resume and source reconnects -- cases where carrying a
        hold across the gap would fire something the user no longer means.
        """
        self._reset_pose_state()
        self.debouncer.reset()
        self.two_hand_debouncer.reset()

    # -- display ----------------------------------------------------------- #

    def build_hud_state(self, now: float) -> HudState:
        status = self.debouncer.status
        two_status = self.two_hand_debouncer.status
        height, width = self._frame_shape or (0, 0)
        stats = getattr(self.detector, "stats", None)
        return HudState(
            fps=self._fps,
            frame_index=self.summary.frames,
            resolution=(width, height),
            source=self.source.description,
            detector=self.detector.describe(),
            inference_ms=getattr(stats, "mean_inference_ms", 0.0),
            pose=self._primary_pose(),
            hands=tuple(self._poses.values()),
            candidate=status.candidate,
            active=status.active,
            armed_fraction=status.armed_fraction,
            held_seconds=status.held_seconds,
            repeats=status.repeats_fired,
            two_hand_candidate=two_status.candidate,
            two_hand_active=two_status.active,
            two_hand_armed_fraction=two_status.armed_fraction,
            last_action=self._action_id,
            last_message=self._action_message,
            last_ok=self._action_ok,
            last_action_age=(now - self._action_time) if self._action_time else 0.0,
            banner=self._banner,
            banner_age=now - self._action_time if self._action_time else 99.0,
            banner_ok=self._banner_ok,
            banner_duration=max(0.6, self.config.tracking.repeat_initial_delay + 0.4),
            mode_announce=self._mode_announce,
            mode_announce_age=(
                now - self._mode_announce_time if self._mode_announce_time else 99.0
            ),
            mode_announce_duration=self.config.switching.announce_seconds,
            paused=self.paused,
            dry_run=self.config.actions.dry_run,
            volume=self.readouts.volume,
            brightness=self.readouts.brightness,
            warnings=self._hud_warnings(),
            gesture_set=self.mode_gesture_sets.get(self.current_mode.name),
            action_labels=self.action_labels,
            mode=self.current_mode,
            mode_index=self._mode_index,
            mode_count=self.mode_count,
            modes=self.modes,
            two_hand_set=self.two_hand_set,
        )

    def _hud_warnings(self) -> tuple[str, ...]:
        """Short, actionable notices for the HUD (never the full probe dump)."""
        notices: list[str] = []
        for action_id, subsystem in _LEVEL_ACTIONS:
            backend = getattr(self.registry.get(action_id), "backend", None)
            if backend is not None and not backend.available:
                notices.append(f"{subsystem} control unavailable (run --doctor)")
        if not self.config.actions.allow_keyboard:
            notices.append("keyboard actions disabled")
        if self.config.actions.require_focus_title:
            notices.append(f"keys need focus: {self.config.actions.require_focus_title}")
        return tuple(notices)

    def render(self, frame: np.ndarray, now: float) -> tuple[np.ndarray, HudState]:
        """Draw the HUD onto ``frame`` and remember the result for screenshots.

        The overlay draws in place for speed, which is safe because the frame it
        is given is already private to this iteration: mirroring (on by default)
        returns a new array, and every source produces a fresh frame per read.
        Anything that reuses a buffer would need a copy here.
        """
        state = self.build_hud_state(now)
        display = self.hud.render(frame, state)
        self._last_display = display
        return display, state

    def save_frame(self, frame: np.ndarray | None, directory: str | Path) -> str | None:
        """Write a frame to disk. Used by ``--save-frames`` and the ``s`` key.

        The filename includes a monotonic counter, because a millisecond
        timestamp is not unique when frames are saved faster than that: with
        ``--save-every 1`` on a fast source, several frames land in the same
        millisecond and would otherwise overwrite each other silently.
        """
        if frame is None:
            return None
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            self._saved_count += 1
            target = path / f"frame-{stamp}-{self._saved_count:05d}.png"
            while target.exists():  # a second safety net across runs
                self._saved_count += 1
                target = path / f"frame-{stamp}-{self._saved_count:05d}.png"
            _cv2().imwrite(str(target), frame)
        except Exception as exc:
            _LOG.warning("could not write a frame to %s: %s", path, exc)
            return None
        _LOG.info("saved %s", target)
        return str(target)

    # -- main loop --------------------------------------------------------- #

    def run(
        self,
        *,
        max_frames: int | None = None,
        save_frames: str | Path | None = None,
        save_every: int = 0,
        on_frame: Callable[[np.ndarray, HudState], None] | None = None,
    ) -> RunSummary:
        """Run the control loop until quit, an error, or ``max_frames``.

        The ``finally`` block is the important part: whatever happens inside,
        the camera is released and the window is closed.
        """
        cv2 = _cv2()
        started = self._clock()
        stride = max(1, self.config.detector.stride)
        saved: list[str] = []
        frame_index = 0
        read_failures = 0
        next_detect_frame = 0
        self._save_dir = Path(save_frames) if save_frames is not None else None
        # Asking for frames to be saved and getting nothing is never what was
        # meant, so ``--save-frames`` on its own saves every frame and
        # ``--save-every`` is what throttles it.
        save_stride = max(1, save_every) if save_frames is not None else 0

        try:
            self.open()
            while not self._stopping:
                ok, frame = self.source.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures == 1:
                        _LOG.warning("no frame available from %s", self.source.description)
                    if read_failures > 10:
                        if self.summary.reconnects >= self.config.camera.reopen_attempts:
                            _LOG.error(
                                "giving up after %d reconnection attempt(s)",
                                self.summary.reconnects,
                            )
                            break
                        self.summary.reconnects += 1
                        _LOG.info("reopening source (attempt %d)", self.summary.reconnects)
                        time.sleep(self.config.camera.reopen_delay)
                        if self.source.reopen():
                            read_failures = 0
                            self._opened = True
                            self._reset_pipeline()
                    else:
                        time.sleep(0.01)
                    continue

                read_failures = 0
                now = self._clock()
                if self.config.camera.mirror:
                    frame = cv2.flip(frame, 1)

                detect = frame_index >= next_detect_frame
                if detect:
                    next_detect_frame = frame_index + stride

                self.summary.frames += 1
                self._frame_shape = frame.shape[:2]

                try:
                    self.process(frame, now, detect=detect)
                except DetectorError:
                    raise
                except Exception:
                    _LOG.exception("frame processing failed; continuing")
                    self._warn_once("a frame failed to process (see the log)")

                self.readouts.maybe_refresh(self.registry, now)
                self._update_fps(now)

                if self.config.overlay.show_window or save_frames is not None or on_frame:
                    display, state = self.render(frame, now)
                    if on_frame is not None:
                        on_frame(display, state)
                    if save_stride and frame_index % save_stride == 0:
                        written = self.save_frame(display, save_frames)
                        if written:
                            saved.append(written)
                    if self.config.overlay.show_window:
                        cv2.imshow(self.config.overlay.window_title, display)
                        if not self._handle_keys(cv2.waitKey(1) & 0xFF):
                            break

                frame_index += 1
                if max_frames is not None and frame_index >= max_frames:
                    break
        finally:
            self.summary.duration = self._clock() - started
            self.summary.mean_fps = (
                self.summary.frames / self.summary.duration if self.summary.duration > 0 else 0.0
            )
            self.summary.saved_frames = tuple(saved)
            self.summary.actions_ok = self.registry.total_invocations - self.registry.total_failures
            self.summary.actions_failed = self.registry.total_failures
            self.shutdown()

        return self.summary

    def _update_fps(self, now: float) -> None:
        self._frame_times.append(now)
        # A one-second sliding window: a plain running mean would take many
        # seconds to reflect a stall, which is exactly when the number matters.
        cutoff = now - 1.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.pop(0)
        if len(self._frame_times) > 1:
            span = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def _handle_keys(self, key: int) -> bool:
        """Handle a keypress. Returns False when the loop should stop.

        Both ``p`` and ``space`` toggle the pause: ``p`` is what people reach for,
        and ``space`` was the original binding, so both are accepted rather than
        making anyone learn which one this program chose.
        """
        if key == 255:
            return True
        if key in _KEY_QUIT:
            return False
        if key in _KEY_PAUSE:
            self.set_paused(not self.paused)
            _LOG.info("paused" if self.paused else "resumed")
        elif key in _KEY_MODE_NEXT:
            self._announce(self.step_mode(1))
        elif key in _KEY_MODE_PREV:
            self._announce(self.step_mode(-1))
        elif key in _KEY_MODE_HOME:
            self._announce(self.home_mode())
        elif key == _KEY_DEBUG:
            overlay = self.config.overlay
            overlay.show_timings = not overlay.show_timings
            overlay.show_finger_states = overlay.show_timings
        elif key == _KEY_DRYRUN:
            self.config.actions.dry_run = not self.config.actions.dry_run
            self.registry.dry_run = self.config.actions.dry_run
            _LOG.info("dry run %s", "on" if self.config.actions.dry_run else "off")
        elif key == _KEY_SCREENSHOT:
            self.save_frame(self._last_display, self._save_dir or Path.cwd() / "gesture-frames")
        elif key == _KEY_RESET:
            self._reset_pipeline()
        return True

    def _announce(self, label: str) -> None:
        """Show the mode banner for a change made from the keyboard."""
        self._mode_announce = label
        self._mode_announce_time = self._clock()
