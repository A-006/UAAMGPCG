cmake_minimum_required(VERSION 3.10)
project(UAAMG LANGUAGES CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Subdirectories
add_subdirectory(solvers)   # Ax=b solvers (jacobi, rbgs, cg, vcycle, amgpcg)
add_subdirectory(src)       # 2D fluid simulation (LFM, uses Jacobi for pressure)
add_subdirectory(test)      # Unit tests
