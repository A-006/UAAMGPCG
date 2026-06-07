# LFM simulation library
set(LFM_SOURCES
    core/grid.cpp
    core/grid_3d.cpp
    numerics/advection/advection_3d.cpp
    numerics/bc/patches_3d.cpp
    numerics/pressure/pressure_3d.cpp
    simulator/simulator_3d.cpp
    io/vtk_writer_3d.cpp
    core/scalar_field_3d.cpp
    simulator/scenarios/2d/karman.cpp
    simulator/scenarios/scenario_registry.cpp
    simulator/scenarios/2d/karman_scenario.cpp
    simulator/scenarios/2d/smoke_scenario.cpp
    simulator/scenarios/scenario_3d_registry.cpp
    simulator/scenarios/3d/vortex_ring.cpp
    simulator/scenarios/3d/trefoil_knot.cpp
    simulator/scenarios/3d/delta_wing.cpp
    simulator/scenarios/3d/fire_ball.cpp
    simulator/scenarios/3d/wind_turbine.cpp
    config/config.cpp
    config/cli.cpp
    numerics/advection/advection.cpp
    numerics/bc/patches.cpp
    io/force.cpp
    io/vtk_writer.cpp
    numerics/pressure/pressure.cpp
    simulator/chorin_simulator.cpp
    simulator/runner.cpp
    simulator/factory.cpp
    solver/jacobi.cpp
    solver/rbgs.cpp
    solver/pcg.cpp
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
    simulator/flow_map_2d.cpp
    simulator/lfm_simulator.cpp
    simulator/flow_map_3d.cpp
    simulator/lfm_simulator_3d.cpp
)

add_library(lfm_lib ${LFM_SOURCES})

# OpenMP: the LFM flow-map marching / advection are per-cell independent loops
# and were single-threaded — the main CPU bottleneck. Multi-thread them.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lfm_lib PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for lfm_lib")
endif()

# Main executable
add_executable(lfm_2d main.cpp)
target_link_libraries(lfm_2d PRIVATE lfm_lib)

# Ax=b solvers (standalone + MAC-grid Poisson solvers)
add_subdirectory(solver)
