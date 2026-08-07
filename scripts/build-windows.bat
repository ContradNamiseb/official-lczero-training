@echo off
REM Build the C++ data loader natively on Windows with MSVC.
REM
REM Produces build\windows\_lczero_training.cp312-win_amd64.pyd and copies it
REM next to the Python package, so `from lczero_training.dataloader import
REM make_dataloader` works without WSL.
REM
REM Usage (from the repo root):
REM   scripts\build-windows.bat            build (and configure if needed)
REM   scripts\build-windows.bat test       build, then run the C++ tests
REM   scripts\build-windows.bat reconfigure  wipe build\windows and start over
REM
REM Requires Visual Studio with the C++ workload, and the Python 3.12
REM DirectML environment in .venv-directml (see docs/directml_training_port.md).

setlocal enabledelayedexpansion

set "REPO=%~dp0.."
pushd "%REPO%"
set "REPO=%CD%"

set "VENV=%REPO%\.venv-directml"
if not exist "%VENV%\Scripts\python.exe" (
  echo ERROR: %VENV% not found. Create the DirectML environment first:
  echo     uv venv --python 3.12 .venv-directml
  popd & exit /b 1
)

REM Locate Visual Studio. vswhere is not always on PATH, so try it by full
REM path first and fall back to the well-known install location.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSPATH="
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSPATH=%%i"
)
if not defined VSPATH (
  for /d %%d in ("%ProgramFiles%\Microsoft Visual Studio\*") do (
    for /d %%e in ("%%d\*") do (
      if exist "%%e\VC\Auxiliary\Build\vcvars64.bat" set "VSPATH=%%e"
    )
  )
)
if not defined VSPATH (
  echo ERROR: could not find a Visual Studio installation with the C++ workload.
  popd & exit /b 1
)

echo Using Visual Studio at %VSPATH%
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo ERROR: vcvars64.bat failed.
  popd & exit /b 1
)

REM Prepend AFTER vcvars, using delayed expansion: expanding %PATH% on the
REM same line as the call would capture the pre-vcvars value and silently
REM drop the MSVC toolchain from PATH.
set "PATH=%VENV%\Scripts;!PATH!"
set "PKG_CONFIG_PATH=%VENV%\Lib\site-packages\pybind11\share\pkgconfig"

if /i "%~1"=="reconfigure" (
  if exist "build\windows" rmdir /s /q "build\windows"
)

if not exist "build\windows\build.ninja" (
  echo Configuring...
  meson setup build\windows --native-file native-windows.ini --buildtype=release
  if errorlevel 1 ( popd & exit /b 1 )
)

echo Building...
meson compile -C build\windows
if errorlevel 1 ( popd & exit /b 1 )

echo Installing extension into src\lczero_training...
copy /y "build\windows\_lczero_training.cp312-win_amd64.pyd" "src\lczero_training\_lczero_training.pyd" >nul
if errorlevel 1 ( popd & exit /b 1 )

if /i "%~1"=="test" (
  echo Running C++ tests...
  meson test -C build\windows --print-errorlogs
  if errorlevel 1 ( popd & exit /b 1 )
)

echo.
echo Done. Verify with:
echo     .venv-directml\Scripts\python.exe -c "from lczero_training.dataloader import make_dataloader; print('ok')"
popd
exit /b 0
