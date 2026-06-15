find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS
        /usr/include
        /usr/local/include
        ${TENSORRT_ROOT}/include
        $ENV{TENSORRT_ROOT}/include
    PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
)

find_library(NVINFER_LIBRARY
    NAMES nvinfer
    PATHS
        /usr/lib
        /usr/local/lib
        ${TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib
    PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
)

find_library(NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    PATHS
        /usr/lib
        /usr/local/lib
        ${TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib
    PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
)

find_library(NVPARSERS_LIBRARY
    NAMES nvparsers
    PATHS
        /usr/lib
        /usr/local/lib
        ${TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib
    PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
)

find_library(NVONNXPARSER_LIBRARY
    NAMES nvonnxparser
    PATHS
        /usr/lib
        /usr/local/lib
        ${TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib
    PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu
)

set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
set(TENSORRT_LIBRARIES
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    ${NVPARSERS_LIBRARY}
    ${NVONNXPARSER_LIBRARY}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        NVINFER_LIBRARY
        TENSORRT_INCLUDE_DIR
    VERSION_VAR TENSORRT_VERSION
)

mark_as_advanced(
    TENSORRT_INCLUDE_DIR
    NVINFER_LIBRARY
    NVINFER_PLUGIN_LIBRARY
    NVPARSERS_LIBRARY
    NVONNXPARSER_LIBRARY
)
