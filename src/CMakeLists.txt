# LFM simulation library
set(LFM_SOURCES
    core/grid.cpp
    core/grid_3d.cpp
    advection/advection_3d.cpp
    bc/patches_3d.cpp
    pressure/pressure_3d.cpp
    simulator/simulator_3d.cpp
    io/vtk_writer_3d.cpp
    scalar/scalar_field_3d.cpp
    scenarios/karman.cpp
    scenarios/vortex_ring.cpp
    scenarios/trefoil_knot.cpp
    scenarios/delta_wing.cpp
    scenarios/fire_ball.cpp
    scenarios/wind_turbine.cpp
    config/config.cpp
    advection/advection.cpp
    boundary/boundary.cpp
    bc/patches.cpp
    force/force.cpp
    io/vtk_writer.cpp
    pressure/pressure.cpp
    simulator/simulator.cpp
    simulator/factory.cpp
    solver/jacobi.cpp
    solver/rbgs.cpp
    solver/pcg.cpp
    solver/jacobi_3d.cpp
    solver/rbgs_3d.cpp
    solver/pcg_3d.cpp
    solver/factory_3d.cpp
    solver/preconditioner/gmg_preconditioner.cpp
    solver/preconditioner/amg_preconditioner.cpp
    solver/preconditioner/uaamg_preconditioner.cpp
    solver/preconditioner/gmg_preconditioner_3d.cpp
    solver/preconditioner/amg_preconditioner_3d.cpp
    solver/preconditioner/uaamg_preconditioner_3d.cpp
    solver/factory.cpp
    lfm/flow_map_2d.cpp
    lfm/lfm_simulator.cpp
)

add_library(lfm_lib ${LFM_SOURCES})

# Main executable
add_executable(lfm_2d main.cpp)
target_link_libraries(lfm_2d PRIVATE lfm_lib)

# Ax=b solvers (standalone + MAC-grid Poisson solvers)
add_subdirectory(solver)
