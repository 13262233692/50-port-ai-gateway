#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_DIR/build"

BUILD_TYPE="Release"
USE_NVDEC="ON"
USE_TENSORRT="ON"
JETSON_PLATFORM="OFF"
CLEAN_BUILD=0

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --debug          Build debug version"
    echo "  --release        Build release version (default)"
    echo "  --no-nvdec       Disable NVDEC hardware decoding"
    echo "  --no-tensorrt    Disable TensorRT inference"
    echo "  --jetson         Build for Jetson platform"
    echo "  --clean          Clean build directory before building"
    echo "  --help           Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --no-nvdec)
            USE_NVDEC="OFF"
            shift
            ;;
        --no-tensorrt)
            USE_TENSORRT="OFF"
            shift
            ;;
        --jetson)
            JETSON_PLATFORM="ON"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ "$CLEAN_BUILD" -eq 1 ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

echo "=========================================="
echo "  Port AI Gateway - Build Configuration"
echo "=========================================="
echo "Build type: $BUILD_TYPE"
echo "NVDEC: $USE_NVDEC"
echo "TensorRT: $USE_TENSORRT"
echo "Jetson platform: $JETSON_PLATFORM"
echo "Build directory: $BUILD_DIR"
echo "=========================================="

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DUSE_NVDEC="$USE_NVDEC" \
    -DUSE_TENSORRT="$USE_TENSORRT" \
    -DJETSON_PLATFORM="$JETSON_PLATFORM" \
    -DCMAKE_MODULE_PATH="$PROJECT_DIR/cmake"

echo ""
echo "Building..."
echo ""

make -j$(nproc)

echo ""
echo "Build completed successfully!"
echo "Binary: $BUILD_DIR/bin/port_ai_gateway"
