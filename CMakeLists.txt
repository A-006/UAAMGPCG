cmake_minimum_required(VERSION 3.10)
project(UAAMG LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Global include path
include_directories(${CMAKE_SOURCE_DIR}/include)

# Subdirectories
add_subdirectory(src)
add_subdirectory(test)

# CUDA solver (optional — requires nvcc and GPU)
# Build with: CUDACXX=/usr/local/cuda/bin/nvcc cmake ..
if(DEFINED ENV{CUDACXX} OR DEFINED CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
    endif()
    message(STATUS "CUDA enabled — building GPU solver")
    add_subdirectory(src/solver/cuda)
else()
    message(STATUS "CUDA not enabled — skipping GPU solver")
    message(STATUS "  To build with CUDA: CUDACXX=/usr/local/cuda/bin/nvcc cmake ..")
endif()
