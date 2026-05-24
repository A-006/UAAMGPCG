# LFM simulation library
set(LFM_SOURCES
    core/grid.cpp
    core/grid_3d.cpp
    config/config.cpp
    advection/advection.cpp
    boundary/boundary.cpp
    io/vtk_writer.cpp
    pressure/pressure.cpp
    simulator/simulator.cpp
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
)

add_library(lfm_lib ${LFM_SOURCES})

# Main executable
add_executable(lfm_2d main.cpp)
target_link_libraries(lfm_2d PRIVATE lfm_lib)

# Ax=b solvers (standalone + MAC-grid Poisson solvers)
add_subdirectory(solver)
