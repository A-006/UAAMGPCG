# ── LFM 2D Fluid Simulation (library) ──
set(LFM_SOURCES
    lfm/grid.cpp
    lfm/advection.cpp
    lfm/poisson_jacobi.cpp
    lfm/poisson_rbgs.cpp
    lfm/poisson_cg.cpp
    lfm/poisson_pcg.cpp
    lfm/pressure.cpp
    lfm/boundary.cpp
    lfm/vtk_io.cpp
)

add_library(lfm_lib ${LFM_SOURCES})

# ── Main executable ──
add_executable(lfm_2d main.cpp)
target_link_libraries(lfm_2d PRIVATE lfm_lib)

# ── Ax=b solvers ──
add_subdirectory(solver)
