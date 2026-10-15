"""Launcher: find a Python that has the dependencies, then run the controller.

Why this exists
---------------
This machine has six Python installations on ``PATH`` and the default one
(Python 3.14) has none of the dependencies, because MediaPipe publishes no
wheels for it. Running ``python main.py`` therefore fails with a traceback that
names a *module*, which sends you looking for a missing package when the real
problem is a mismatched interpreter.

This script:

* checks whether the *current* interpreter can import the dependencies and runs
  immediately if so;
* otherwise asks the Windows launcher (``py -0p``) and the ``PATH`` for other
  interpreters, tests each one, and re-launches under the first that works;
* prints the interpreter it picked and why, so the choice is never silent;
* falls back to a clear explanation instead of a traceback when nothing works.

Usage is identical to ``python -m gesture_control``::

    python run.py --doctor
    python run.py --preset media
    python run.py                # normal use
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Modules needed for the controller to run at all.
REQUIRED = ("numpy", "cv2", "mediapipe")

#: MediaPipe wheel availability window.
MIN_VERSION = (3, 9)
MAX_VERSION = (3, 12)

_PROBE = (
    "import importlib.util as u, sys;"
    f"missing=[m for m in {REQUIRED!r} if u.find_spec(m) is None];"
    "print('OK' if not missing else 'MISSING:'+','.join(missing))"
)


def probe(executable: str) -> tuple[bool, str]:
    """Return ``(usable, description)`` for a candidate interpreter."""
    try:
        result = subprocess.run(
            [executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (result.stdout or "").strip()
    if output == "OK":
        return True, "all dependencies present"
    if output.startswith("MISSING:"):
        return False, f"missing {output.split(':', 1)[1]}"
    error = (result.stderr or "").strip().splitlines()
    return False, error[-1][:120] if error else f"exit code {result.returncode}"


def candidate_interpreters() -> list[str]:
    """Interpreters to try, in preference order, excluding the current one."""
    seen: list[str] = []
    current = Path(sys.executable).resolve()

    def add(path: str | None) -> None:
        if not path:
            return
        resolved = Path(path)
        try:
            if resolved.resolve() == current:
                return
        except OSError:
            return
        text = str(resolved)
        if text not in seen and resolved.is_file():
            seen.append(text)

    # The Windows launcher knows about every registered installation, and lists
    # them newest-first; reordering by version is handled by the caller.
    launcher = shutil.which("py")
    if launcher:
        try:
            listing = subprocess.run(
                [launcher, "-0p"], capture_output=True, text=True, timeout=15, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        for line in listing.splitlines():
            match = re.search(r"([A-Za-z]:\\[^\s].*?python\.exe)\s*$", line.strip())
            if match:
                add(match.group(1))

    for name in ("python3.12", "python3.11", "python3.10", "python3.9", "python3", "python"):
        add(shutil.which(name))

    # Common per-user install locations, in case nothing is on PATH.
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    if local.is_dir():
        for directory in sorted(local.iterdir(), reverse=True):
            add(str(directory / "python.exe"))

    return seen


def version_of(executable: str) -> tuple[int, ...]:
    try:
        result = subprocess.run(
            [executable, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return tuple(int(part) for part in result.stdout.strip().split("."))
    except Exception:
        return ()


def in_supported_range(version: tuple[int, ...]) -> bool:
    return bool(version) and MIN_VERSION <= version[:2] <= MAX_VERSION


def find_interpreter(*, want_in_range: bool) -> tuple[str | None, list[str]]:
    """Look for another interpreter that has the dependencies.

    Args:
        want_in_range: when True, only accept interpreters inside the supported
            version window. When False, any interpreter that can import the
            dependencies will do.

    Returns:
        ``(chosen_or_None, report_lines)``.
    """
    report: list[str] = []
    for candidate in candidate_interpreters():
        candidate_version = version_of(candidate)
        if want_in_range and not in_supported_range(candidate_version):
            shown = ".".join(str(part) for part in candidate_version) or "unknown"
            report.append(
                f"  skip  {candidate}  (Python {shown}, outside the supported "
                f"{MIN_VERSION[0]}.{MIN_VERSION[1]}-{MAX_VERSION[0]}.{MAX_VERSION[1]} range)"
            )
            continue
        ok, why = probe(candidate)
        report.append(f"  {'use ' if ok else 'skip'}  {candidate}  ({why})")
        if ok:
            return candidate, report
    return None, report


def _relaunch(interpreter: str, reason: str) -> int:
    print(f"run.py: {reason}\nrun.py: re-launching with {interpreter}\n", file=sys.stderr)
    # Replace this process so signals, exit codes and Ctrl+C behave normally.
    return subprocess.run([interpreter, str(Path(__file__)), *sys.argv[1:]], check=False).returncode


def _run_here() -> int:
    from gesture_control.cli import main as cli_main

    return cli_main()


def main() -> int:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    usable, description = probe(sys.executable)
    version = sys.version_info[:2]
    supported = in_supported_range(version)

    # Best case: dependencies present and the version is one the project is
    # tested against.
    if usable and supported:
        return _run_here()

    if usable and not supported:
        # The dependencies import here, so it does work -- but the version is
        # outside the tested range. Prefer a supported interpreter if one is
        # available, and otherwise carry on rather than refusing to run.
        chosen, report = find_interpreter(want_in_range=True)
        if chosen is not None:
            for line in report:
                print(line, file=sys.stderr)
            return _relaunch(
                chosen,
                f"Python {version[0]}.{version[1]} is outside the supported "
                f"{MIN_VERSION[0]}.{MIN_VERSION[1]}-{MAX_VERSION[0]}.{MAX_VERSION[1]} range",
            )
        print(
            f"run.py: note: Python {version[0]}.{version[1]} is outside the range this "
            f"project is tested against\n"
            f"        ({MIN_VERSION[0]}.{MIN_VERSION[1]}-{MAX_VERSION[0]}.{MAX_VERSION[1]}, "
            f"the versions MediaPipe publishes wheels for), but the dependencies import\n"
            f"        and it will be used. Report anything strange with --doctor.\n",
            file=sys.stderr,
        )
        return _run_here()

    print(
        f"run.py: this interpreter cannot run the controller\n"
        f"        {sys.executable}\n"
        f"        Python {version[0]}.{version[1]}, {description}",
        file=sys.stderr,
    )

    chosen, report = find_interpreter(want_in_range=True)
    if chosen is None:
        # Nothing in range: accept any interpreter that has the dependencies.
        fallback, fallback_report = find_interpreter(want_in_range=False)
        report.extend(fallback_report)
        chosen = fallback

    if chosen is None:
        print("run.py: looking for another interpreter with the dependencies...", file=sys.stderr)
        for line in report:
            print(line, file=sys.stderr)
        print(
            "\nrun.py: no suitable interpreter was found.\n"
            "Install the dependencies, then run this script again:\n"
            f'    "{sys.executable}" -m pip install -r requirements.txt\n'
            f"Python {MIN_VERSION[0]}.{MIN_VERSION[1]} - "
            f"{MAX_VERSION[0]}.{MAX_VERSION[1]} is recommended "
            f"(MediaPipe has no wheels outside that range).",
            file=sys.stderr,
        )
        return 1

    for line in report:
        print(line, file=sys.stderr)
    return _relaunch(chosen, "this interpreter is missing the dependencies")


if __name__ == "__main__":
    raise SystemExit(main())
