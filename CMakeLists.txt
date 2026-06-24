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

# Optional NVIDIA AMGX Poisson backend (solver=amgx). Uses AMGX's C API only,
# so it links the prebuilt libamgxsh.so without needing nvcc. See src/CMakeLists.txt.
option(UAAMG_WITH_AMGX "Build the AMGX Poisson solver backend" OFF)
set(AMGX_ROOT "" CACHE PATH "Path to AMGX (with include/amgx_c.h and build/libamgxsh.so)")

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

    # GPU Cylinder benchmark (2D)
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

    # GPU Cylinder benchmark (3D)
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

    # GPU-solver LFM Cylinder runner (CPU sim loop, GPU Poisson solve)
    add_executable(run_karman_long_gpu tools/run_karman_long_gpu.cu)
    set_target_properties(run_karman_long_gpu PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(run_karman_long_gpu PRIVATE lfm_lib cuda_uaamg_lib cuda_cg_lib)
    target_include_directories(run_karman_long_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # ── Unified INPUT-file-driven launcher for ALL GPU 3D LFM scenes ──
    # One binary replaces the former per-scene runners (run_vortex_ring_*,
    # run_*_collision_*, run_delta_wing_*, run_collision_paper, plus the CPU
    # reconnection/trefoil demos): the scenario + every parameter come from an
    # INI file (see inputs/*.in). Scenarios: vortex_ring | vortex_collision |
    # collision_paper | delta_wing | vortex_reconnection | trefoil_knot.
    #   ./build/cfdsim inputs/vortex_collision.in
    # collision_paper's author cross-check (raw IC load, velocity dump) lives on
    # as the ic_dir= / dump_vel= knobs.
    # CUDA build: compile the .cpp as CUDA (nvcc) so it can drive the GPU-resident
    # simulator. HAVE_CUDA enables the GPU backend; runs on GPU by default
    # (backend=auto), with a CPU fallback when no device is present.
    # main.cpp is just main(); make_simulator_3d.cpp builds the GPU/CPU
    # simulator, so it needs nvcc too. Both compile as CUDA in this build.
    add_executable(cfdsim src/main.cpp src/integrator/make_simulator_3d.cpp)
    set_source_files_properties(src/main.cpp src/integrator/make_simulator_3d.cpp
        PROPERTIES LANGUAGE CUDA)
    set_target_properties(cfdsim PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_compile_definitions(cfdsim PRIVATE HAVE_CUDA)
    target_link_libraries(cfdsim PRIVATE cuda_lfm_3d_lib cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
    target_include_directories(cfdsim PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # GPU Cylinder validate (Cd/Cl/St)
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
    message(STATUS "CUDA not enabled — building cfdsim as a CPU-only binary")
    message(STATUS "  To build with GPU support: CUDACXX=/usr/local/cuda/bin/nvcc cmake ..")

    # CPU-only cfdsim: the SAME sources, compiled by the C++ compiler (no nvcc).
    # Without HAVE_CUDA make_simulator_3d.cpp pulls in no CUDA headers, so
    # backend=auto/gpu both resolve to the host LFMSimulator3D. GPU optional.
    add_executable(cfdsim src/main.cpp src/integrator/make_simulator_3d.cpp)
    target_link_libraries(cfdsim PRIVATE lfm_lib)
    target_include_directories(cfdsim PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()
