cmake_minimum_required(VERSION 3.10)

# SET(CMAKE_BUILD_TYPE "Release")
project(pso_cpu LANGUAGES CXX CUDA)

option(BUILD_WITH_CUDA "Build with NVidia CUDA" OFF)

mark_as_advanced(BUILD_WITH_CUDA)
if(BUILD_WITH_CUDA)
    add_definitions(-DBUILD_WITH_CUDA)
endif(BUILD_WITH_CUDA)

if(BUILD_WITH_CUDA)
    # Figure out CUDA version
    if(CMAKE_CUDA_COMPILER_LOADED)
        if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
            set(CUDA_VERSION "${CMAKE_MATCH_1}")
        endif()
    endif()
    # CUDA flags
    # set(CMAKE_CUDA_ARCHITECTURES 75)
    # set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    project(pso_cuda LANGUAGES CXX CUDA)
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)
    # Check CUDA version
    if (CUDAToolkit_VERSION VERSION_LESS "11.7")
        message(FATAL_ERROR "This project requires CUDA 11.7 or higher")
    endif ()
endif(BUILD_WITH_CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 75)
SET(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
# set(CUDA_NVCC_FLAGS "-g -G")

include_directories(include)

add_executable(pso_cuda src/kernels.cu src/main.cpp)
add_executable(pso_cpu src/mainCPU.cpp)           