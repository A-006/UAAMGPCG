# LFM simulation library
set(LFM_SOURCES
    mesh/grid.cpp
    mesh/grid_3d.cpp
    integrator/ops/advection_3d.cpp
    mesh/bc/patches_3d.cpp
    integrator/ops/pressure_3d.cpp
    integrator/simulator_3d.cpp
    integrator/scene_3d.cpp
    integrator/runner_3d.cpp
    io/vtk_writer_3d.cpp
    io/vtk_slim_3d.cpp
    core/scalar_field_3d.cpp
    io/ic/2d/karman.cpp
    io/ic/2d/leapfrog.cpp
    io/ic/2d/primitives_2d.cpp
    io/ic/2d/generic_scenario.cpp
    io/ic/scenario_registry.cpp
    io/ic/3d/vortex_ring.cpp
    io/ic/3d/trefoil_knot.cpp
    io/ic/3d/delta_wing.cpp
    io/ic/3d/fire_ball.cpp
    io/ic/3d/wind_turbine.cpp
    core/config.cpp
    io/cli.cpp
    integrator/ops/advection.cpp
    mesh/bc/patches.cpp
    io/force.cpp
    io/vtk_writer.cpp
    integrator/ops/pressure.cpp
    integrator/chorin/chorin_simulator.cpp
    integrator/runner.cpp
    integrator/factory.cpp
    solver/jacobi.cpp
    solver/rbgs.cpp
    solver/pcg.cpp
    solver/amgx_solver.cpp
    solver/amgx_solver_3d.cpp
    solver/amgx_backend.cpp
    solver/jacobi_3d.cpp
    solver/rbgs_3d.cpp
    solver/pcg_3d.cpp
    solver/factory_3d.cpp
    solver/preconditioner/2d/gmg_preconditioner.cpp
    solver/preconditioner/2d/amg_preconditioner.cpp
    solver/preconditioner/2d/uaamg_preconditioner.cpp
    solver/preconditioner/3d/gmg_preconditioner_3d.cpp
    solver/preconditioner/3d/amg_preconditioner_3d.cpp
    solver/preconditioner/3d/uaamg_preconditioner_3d.cpp
    solver/factory.cpp
    integrator/lfm/flow_map_2d.cpp
    integrator/lfm/lfm_simulator.cpp
    integrator/lfm/flow_map_3d.cpp
    integrator/lfm/lfm_simulator_3d.cpp
)

add_library(lfm_lib ${LFM_SOURCES})

# OpenMP: the LFM flow-map marching / advection are per-cell independent loops
# and were single-threaded — the main CPU bottleneck. Multi-thread them.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lfm_lib PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for lfm_lib")
endif()

# ── Optional NVIDIA AMGX Poisson backend (solver=amgx) ──
# AMGX is reached through its plain C API (amgx_c.h), so amgx_solver.cpp builds
# with the host C++ compiler — no nvcc needed here. We only link the prebuilt
# libamgxsh.so plus the CUDA runtime/cublas it depends on. Enable with:
#   cmake -DUAAMG_WITH_AMGX=ON -DAMGX_ROOT=/path/to/AMGX ..
# AMGX_ROOT must contain include/amgx_c.h and build/libamgxsh.so.
if(UAAMG_WITH_AMGX)
    if(NOT AMGX_ROOT)
        message(FATAL_ERROR "UAAMG_WITH_AMGX=ON requires -DAMGX_ROOT=/path/to/AMGX")
    endif()
    find_path(AMGX_INCLUDE_DIR amgx_c.h HINTS "${AMGX_ROOT}/include" REQUIRED)
    find_library(AMGX_LIB amgxsh HINTS "${AMGX_ROOT}/build" "${AMGX_ROOT}/lib" REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(lfm_lib PRIVATE HAVE_AMGX)
    target_include_directories(lfm_lib PRIVATE "${AMGX_INCLUDE_DIR}")
    target_link_libraries(lfm_lib PUBLIC "${AMGX_LIB}" CUDA::cudart CUDA::cublas)
    message(STATUS "AMGX backend enabled: ${AMGX_LIB}")
else()
    message(STATUS "AMGX backend disabled (set -DUAAMG_WITH_AMGX=ON -DAMGX_ROOT=... to enable)")
endif()

# The single program is `cfdsim` (defined in the top-level CMakeLists.txt). It
# handles both 2D (CPU) and 3D (GPU/CPU) scenarios, so the former `lfm_2d`
# entry point (src/main.cpp) has been removed.

# Ax=b solvers (standalone + MAC-grid Poisson solvers)
add_subdirectory(solver)
