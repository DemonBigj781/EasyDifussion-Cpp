@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "LLAMA_SOURCE=%PROJECT_ROOT%\source\llama.cpp"
set "LLAMA_BUILD_DIR=%LLAMA_SOURCE%\build-windows-x64"
set "SDKIT_SOURCE=%PROJECT_ROOT%\source\sdkit3-port-source"
set "SDKIT_BUILD_DIR=%SDKIT_SOURCE%\build\windows-x64-cpu"
set "SDKIT_TARGET_DIR=%PROJECT_ROOT%\backends\sdkit3\windows-x64-cpu-any"
set "GGUF_ENV=%PROJECT_ROOT%\.venv"

set BUILD_NATIVE=0
set BUILD_LLAMA=0
set INSTALL_GGUF=0
set BUILD_PLATFORM=cpu
set JOB_COUNT=2

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--all" (
  set BUILD_NATIVE=1
  set BUILD_LLAMA=1
  set INSTALL_GGUF=1
  shift
  goto parse
)
if /I "%~1"=="--native-build" (
  set BUILD_NATIVE=1
  shift
  goto parse
)
if /I "%~1"=="--llama-build" (
  set BUILD_LLAMA=1
  shift
  goto parse
)
if /I "%~1"=="--gguf-tools" (
  set INSTALL_GGUF=1
  shift
  goto parse
)
if /I "%~1"=="--cpu" (
  set BUILD_PLATFORM=cpu
  shift
  goto parse
)
if /I "%~1"=="--jobs" (
  if "%~2"=="" goto jobs_error
  set "JOB_COUNT=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--help" goto help
if /I "%~1"=="-h" goto help

echo ERROR: Unknown option: %~1
exit /b 1

:jobs_error
echo ERROR: --jobs requires a positive integer
exit /b 1

:parsed
if "%BUILD_NATIVE%%BUILD_LLAMA%%INSTALL_GGUF%"=="000" (
  set BUILD_LLAMA=1
  set INSTALL_GGUF=1
)

if not exist "%PROJECT_ROOT%\LICENSE" (
  echo ERROR: The project LICENSE file is missing.
  exit /b 1
)
if not exist "%PROJECT_ROOT%\THIRD_PARTY_NOTICES.md" (
  echo ERROR: THIRD_PARTY_NOTICES.md is missing.
  exit /b 1
)
if not exist "%LLAMA_SOURCE%\CMakeLists.txt" (
  echo ERROR: The vendored llama.cpp source is missing.
  exit /b 1
)
if not exist "%SDKIT_SOURCE%\CMakeLists.txt" (
  echo ERROR: The vendored sdkit3 source is missing.
  exit /b 1
)

where cmake >nul 2>nul
if errorlevel 1 (
  echo ERROR: cmake is required.
  exit /b 1
)

where ninja >nul 2>nul
if errorlevel 1 (
  echo ERROR: ninja is required for the Windows installer.
  exit /b 1
)

if "%BUILD_LLAMA%"=="1" (
  echo Configuring llama.cpp for Windows x64 CPU...
  cmake -S "%LLAMA_SOURCE%" -B "%LLAMA_BUILD_DIR%" -G Ninja ^
    -DGGML_CUDA=OFF ^
    -DGGML_SYCL=OFF ^
    -DLLAMA_CURL=OFF ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=ON ^
    -DLLAMA_BUILD_SERVER=ON ^
    -DCMAKE_BUILD_TYPE=Release
  if errorlevel 1 exit /b 1

  cmake --build "%LLAMA_BUILD_DIR%" --target llama-cli llama-server llama-quantize --parallel %JOB_COUNT%
  if errorlevel 1 exit /b 1
)

if "%BUILD_NATIVE%"=="1" (
  echo Configuring sdkit/stable-diffusion.cpp for Windows x64 CPU...
  cmake -S "%SDKIT_SOURCE%" -B "%SDKIT_BUILD_DIR%" -G Ninja ^
    -DSD_CUDA=OFF ^
    -DSD_SYCL=OFF ^
    -DSDKIT_BUILD_LLAMA_RUNTIME=ON ^
    -DSDKIT_LLAMA_BUILD_JOBS=%JOB_COUNT% ^
    -DSDKIT_BUILD_NATIVE_VISION=OFF ^
    -DCMAKE_BUILD_TYPE=Release
  if errorlevel 1 exit /b 1

  cmake --build "%SDKIT_BUILD_DIR%" --target sdkit --parallel %JOB_COUNT%
  if errorlevel 1 exit /b 1

  if not exist "%SDKIT_TARGET_DIR%" mkdir "%SDKIT_TARGET_DIR%"
  xcopy /E /I /Y "%SDKIT_BUILD_DIR%\bin\*" "%SDKIT_TARGET_DIR%\" >nul
  if errorlevel 1 exit /b 1
  echo Native bundle ready: %SDKIT_TARGET_DIR%
)

if "%INSTALL_GGUF%"=="1" (
  if not exist "%GGUF_ENV%\Scripts\python.exe" (
    echo ERROR: The main Easy Diffusion venv is missing. Create a Python 3.13 venv at .venv first.
    exit /b 1
  )

  for /f %%V in ('"%GGUF_ENV%\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%V
  if not "%PYTHON_VERSION%"=="3.13" (
    echo ERROR: Easy Diffusion's main venv must use Python 3.13; found %PYTHON_VERSION%.
    exit /b 1
  )

  "%GGUF_ENV%\Scripts\python.exe" -m pip install --upgrade pip wheel
  if errorlevel 1 exit /b 1
  "%GGUF_ENV%\Scripts\python.exe" -m pip install --editable "%LLAMA_SOURCE%\gguf-py"
  if errorlevel 1 exit /b 1
  "%GGUF_ENV%\Scripts\python.exe" -m pip install "numpy>=2.1,<2.3" "sentencepiece>=0.1.98,<0.3" "transformers==4.57.6" "protobuf>=4.21,<5"
  if errorlevel 1 exit /b 1
  "%GGUF_ENV%\Scripts\python.exe" -c "import google.protobuf, numpy, sentencepiece, transformers, gguf"
  if errorlevel 1 exit /b 1
)

echo Windows native tools are ready.
exit /b 0

:help
echo Usage: install.bat [options]
echo.
echo Options:
echo   --all           Build native bundle, llama.cpp tools, and GGUF tooling.
echo   --native-build  Build sdkit/stable-diffusion.cpp with bundled llama-server.
echo   --llama-build   Build llama-cli, llama-server, and llama-quantize.
echo   --gguf-tools    Install GGUF conversion tools into .venv.
echo   --cpu           Build CPU-only targets. Windows GPU backends can be added later.
echo   --jobs N        Set parallel build job count.
echo   -h, --help      Show this help.
exit /b 0
