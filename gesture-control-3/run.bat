@echo off
rem ---------------------------------------------------------------------------
rem Launcher for Windows.
rem
rem run.py inspects the interpreter it is started with, and re-launches under a
rem suitable one if the dependencies are missing. That matters on a machine with
rem several Python installations, where the default `python` is often too new for
rem MediaPipe and the resulting error names a module rather than the interpreter.
rem
rem   run.bat --doctor         check what this machine supports
rem   run.bat --self-test      verify gesture recognition
rem   run.bat                  normal use
rem ---------------------------------------------------------------------------
setlocal
set "HERE=%~dp0"

if exist "%HERE%.venv\Scripts\python.exe" (
    "%HERE%.venv\Scripts\python.exe" "%HERE%run.py" %*
    exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py "%HERE%run.py" %*
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%HERE%run.py" %*
    exit /b %ERRORLEVEL%
)

echo.
echo No Python interpreter was found on PATH.
echo Install Python 3.12 from https://www.python.org/downloads/ and make sure
echo "Add python.exe to PATH" is ticked during setup.
echo.
pause
exit /b 1
