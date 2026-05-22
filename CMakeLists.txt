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
