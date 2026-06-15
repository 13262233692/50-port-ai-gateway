cmake_minimum_required(VERSION 3.18)
project(port_ai_gateway LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(USE_NVDEC "Enable NVIDIA NVDEC hardware acceleration" ON)
option(USE_TENSORRT "Enable TensorRT inference engine" ON)
option(JETSON_PLATFORM "Build for NVIDIA Jetson platform" OFF)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

if(USE_NVDEC)
    add_definitions(-DUSE_NVDEC)
    find_package(FFmpeg REQUIRED COMPONENTS avcodec avformat avutil swscale)
    include_directories(${FFmpeg_INCLUDE_DIRS})
endif()

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    find_package(TensorRT REQUIRED)
    include_directories(${TensorRT_INCLUDE_DIRS})
endif()

if(JETSON_PLATFORM)
    add_definitions(-DJETSON_PLATFORM)
endif()

file(GLOB_RECURSE COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/common/*.cpp
    ${CMAKE_SOURCE_DIR}/include/common/*.h
)

file(GLOB_RECURSE STREAMING_SOURCES
    ${CMAKE_SOURCE_DIR}/src/streaming/*.cpp
    ${CMAKE_SOURCE_DIR}/include/streaming/*.h
)

file(GLOB_RECURSE INFERENCE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/inference/*.cpp
    ${CMAKE_SOURCE_DIR}/src/inference/*.cu
    ${CMAKE_SOURCE_DIR}/include/inference/*.h
    ${CMAKE_SOURCE_DIR}/include/inference/*.cuh
)

file(GLOB_RECURSE TRACKING_SOURCES
    ${CMAKE_SOURCE_DIR}/src/tracking/*.cpp
    ${CMAKE_SOURCE_DIR}/include/tracking/*.h
)

file(GLOB_RECURSE SAFETY_SOURCES
    ${CMAKE_SOURCE_DIR}/src/safety/*.cpp
    ${CMAKE_SOURCE_DIR}/include/safety/*.h
)

file(GLOB_RECURSE APP_SOURCES
    ${CMAKE_SOURCE_DIR}/src/app/*.cpp
    ${CMAKE_SOURCE_DIR}/include/app/*.h
)

cuda_add_library(port_ai_common STATIC ${COMMON_SOURCES})
target_link_libraries(port_ai_common
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

if(USE_NVDEC)
    cuda_add_library(port_ai_streaming STATIC ${STREAMING_SOURCES})
    target_link_libraries(port_ai_streaming
        port_ai_common
        ${FFmpeg_LIBRARIES}
        ${CUDA_LIBRARIES}
        nvcuvid
        cuda
    )
endif()

if(USE_TENSORRT)
    cuda_add_library(port_ai_inference STATIC ${INFERENCE_SOURCES})
    target_link_libraries(port_ai_inference
        port_ai_common
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
        cudart
    )
    set_target_properties(port_ai_inference PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    cuda_add_library(port_ai_tracking STATIC ${TRACKING_SOURCES})
    target_link_libraries(port_ai_tracking
        port_ai_common
        port_ai_inference
        ${CUDA_LIBRARIES}
    )

    cuda_add_library(port_ai_safety STATIC ${SAFETY_SOURCES})
    target_link_libraries(port_ai_safety
        port_ai_common
        port_ai_tracking
        port_ai_inference
        ${CUDA_LIBRARIES}
    )
endif()

cuda_add_executable(port_ai_gateway
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${APP_SOURCES}
)

target_link_libraries(port_ai_gateway
    port_ai_common
    $<$<BOOL:${USE_NVDEC}>:port_ai_streaming>
    $<$<BOOL:${USE_TENSORRT}>:port_ai_inference>
    $<$<BOOL:${USE_TENSORRT}>:port_ai_tracking>
    $<$<BOOL:${USE_TENSORRT}>:port_ai_safety>
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS port_ai_gateway DESTINATION bin)
