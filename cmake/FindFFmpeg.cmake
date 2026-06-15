find_path(FFMPEG_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS
        /usr/include
        /usr/local/include
        ${FFMPEG_ROOT}/include
        $ENV{FFMPEG_ROOT}/include
    PATH_SUFFIXES ffmpeg
)

find_library(AVCODEC_LIBRARY
    NAMES avcodec
    PATHS
        /usr/lib
        /usr/local/lib
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
)

find_library(AVFORMAT_LIBRARY
    NAMES avformat
    PATHS
        /usr/lib
        /usr/local/lib
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
)

find_library(AVUTIL_LIBRARY
    NAMES avutil
    PATHS
        /usr/lib
        /usr/local/lib
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
)

find_library(SWSCALE_LIBRARY
    NAMES swscale
    PATHS
        /usr/lib
        /usr/local/lib
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
)

find_library(SWRESAMPLE_LIBRARY
    NAMES swresample
    PATHS
        /usr/lib
        /usr/local/lib
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
)

find_library(AVDEVICE_LIBRARY
    NAMES avdevice
    PATHS
        /usr/lib
        /usr/local/lib
        ${FFMPEG_ROOT}/lib
        $ENV{FFMPEG_ROOT}/lib
)

set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})
set(FFMPEG_LIBRARIES
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg
    REQUIRED_VARS
        AVCODEC_LIBRARY
        AVFORMAT_LIBRARY
        AVUTIL_LIBRARY
        SWSCALE_LIBRARY
        FFMPEG_INCLUDE_DIR
    VERSION_VAR FFMPEG_VERSION
)

mark_as_advanced(
    FFMPEG_INCLUDE_DIR
    AVCODEC_LIBRARY
    AVFORMAT_LIBRARY
    AVUTIL_LIBRARY
    SWSCALE_LIBRARY
    SWRESAMPLE_LIBRARY
    AVDEVICE_LIBRARY
)
