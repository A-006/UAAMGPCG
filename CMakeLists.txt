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

    # GPU unit / benchmark tests (moved out of src/solver/cuda)
    add_subdirectory(test/cuda)

    # GPU Karman benchmark (2D)
    add_executable(test_karman_bench_gpu test/integration/test_karman_bench_gpu.cu)
    set_target_properties(test_karman_bench_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(test_karman_bench_gpu PRIVATE lfm_lib cuda_uaamg_lib cuda_cg_lib)
    target_include_directories(test_karman_bench_gpu PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test
    )

    # GPU Karman benchmark (3D)
    add_executable(test_karman_bench_3d_gpu test/integration/test_karman_bench_3d_gpu.cu)
    set_target_properties(test_karman_bench_3d_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(test_karman_bench_3d_gpu PRIVATE lfm_lib cuda_uaamg_lib cuda_cg_lib)
    target_include_directories(test_karman_bench_3d_gpu PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test
    )

    # GPU-solver LFM Karman runner (CPU sim loop, GPU Poisson solve)
    add_executable(run_karman_long_gpu tools/run_karman_long_gpu.cu)
    set_target_properties(run_karman_long_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_karman_long_gpu PRIVATE lfm_lib cuda_uaamg_lib cuda_cg_lib)
    target_include_directories(run_karman_long_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # GPU-solver 3D LFM vortex-ring runner (CPU flow-map loop, GPU 3D Poisson)
    add_executable(run_vortex_ring_lfm_gpu tools/run_vortex_ring_lfm_gpu.cu)
    set_target_properties(run_vortex_ring_lfm_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_vortex_ring_lfm_gpu PRIVATE lfm_lib cuda_uaamg_3d_lib cuda_cg_3d_lib)
    target_include_directories(run_vortex_ring_lfm_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # GPU-solver 3D LFM head-on vortex-ring collision (paper Fig. 3)
    add_executable(run_vortex_ring_collision_lfm_gpu tools/run_vortex_ring_collision_lfm_gpu.cu)
    set_target_properties(run_vortex_ring_collision_lfm_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_vortex_ring_collision_lfm_gpu PRIVATE lfm_lib cuda_uaamg_3d_lib cuda_cg_3d_lib)
    target_include_directories(run_vortex_ring_collision_lfm_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # Fully GPU-resident 3D LFM vortex-ring runner (whole cycle on device)
    add_executable(run_vortex_ring_lfm_gpu_resident tools/run_vortex_ring_lfm_gpu_resident.cu)
    set_target_properties(run_vortex_ring_lfm_gpu_resident PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_vortex_ring_lfm_gpu_resident PRIVATE cuda_lfm_3d_lib cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
    target_include_directories(run_vortex_ring_lfm_gpu_resident PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # Fully GPU-resident 3D LFM head-on collision (paper Fig. 3)
    add_executable(run_vortex_ring_collision_lfm_gpu_resident tools/run_vortex_ring_collision_lfm_gpu_resident.cu)
    set_target_properties(run_vortex_ring_collision_lfm_gpu_resident PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_vortex_ring_collision_lfm_gpu_resident PRIVATE cuda_lfm_3d_lib cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
    target_include_directories(run_vortex_ring_collision_lfm_gpu_resident PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # GPU delta wing at paper resolution (freestream BC + immersed solid)
    add_executable(run_delta_wing_lfm_gpu tools/run_delta_wing_lfm_gpu.cu)
    set_target_properties(run_delta_wing_lfm_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_delta_wing_lfm_gpu PRIVATE cuda_lfm_3d_lib cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
    target_include_directories(run_delta_wing_lfm_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # Paper-faithful collision: 128x256x256, inviscid + BFECC clamp, n=5
    add_executable(run_collision_paper tools/run_collision_paper.cu)
    set_target_properties(run_collision_paper PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_collision_paper PRIVATE cuda_lfm_3d_lib cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
    target_include_directories(run_collision_paper PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # GPU Karman validate (Cd/Cl/St)
    add_executable(test_karman_validate_gpu test/integration/test_karman_validate_gpu.cu)
    set_target_properties(test_karman_validate_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(test_karman_validate_gpu PRIVATE lfm_lib cuda_uaamg_lib cuda_cg_lib)
    target_include_directories(test_karman_validate_gpu PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/test
    )
else()
    message(STATUS "CUDA not enabled — skipping GPU solver")
    message(STATUS "  To build with CUDA: CUDACXX=/usr/local/cuda/bin/nvcc cmake ..")
endif()
