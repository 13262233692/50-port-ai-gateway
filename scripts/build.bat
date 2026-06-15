@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%
set BUILD_DIR=%PROJECT_DIR%build

set BUILD_TYPE=Release
set USE_NVDEC=ON
set USE_TENSORRT=ON
set CLEAN_BUILD=0

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto parse_args
)
if "%~1"=="--release" (
    set BUILD_TYPE=Release
    shift
    goto parse_args
)
if "%~1"=="--no-nvdec" (
    set USE_NVDEC=OFF
    shift
    goto parse_args
)
if "%~1"=="--no-tensorrt" (
    set USE_TENSORRT=OFF
    shift
    goto parse_args
)
if "%~1"=="--clean" (
    set CLEAN_BUILD=1
    shift
    goto parse_args
)
if "%~1"=="--help" (
    goto usage
)
echo Unknown option: %~1
goto usage

:end_parse

if %CLEAN_BUILD%==1 (
    echo Cleaning build directory...
    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
    )
)

echo ==========================================
echo   Port AI Gateway - Build Configuration
echo ==========================================
echo Build type: %BUILD_TYPE%
echo NVDEC: %USE_NVDEC%
echo TensorRT: %USE_TENSORRT%
echo Build directory: %BUILD_DIR%
echo ==========================================

if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

cd /d "%BUILD_DIR%"

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DUSE_NVDEC=%USE_NVDEC% ^
    -DUSE_TENSORRT=%USE_TENSORRT% ^
    -DCMAKE_MODULE_PATH="%PROJECT_DIR%cmake"

if errorlevel 1 (
    echo CMake configuration failed
    exit /b 1
)

echo.
echo Building...
echo.

cmake --build . --config %BUILD_TYPE%

if errorlevel 1 (
    echo Build failed
    exit /b 1
)

echo.
echo Build completed successfully!
echo Binary: %BUILD_DIR%\bin\%BUILD_TYPE%\port_ai_gateway.exe

goto end

:usage
echo Usage: %~nx0 [options]
echo Options:
echo   --debug          Build debug version
echo   --release        Build release version (default)
echo   --no-nvdec       Disable NVDEC hardware decoding
echo   --no-tensorrt    Disable TensorRT inference
echo   --clean          Clean build directory before building
echo   --help           Show this help message

:end
endlocal
