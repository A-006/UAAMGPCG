# Add test/ and include/ to include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ── Standalone solver tests (use test_common.h, no lfm_lib needed) ──
add_executable(test_jacobi solver/test_jacobi.cpp)
add_executable(test_cg solver/test_cg.cpp)
add_executable(test_vcycle solver/test_vcycle.cpp)
add_executable(test_amgpcg solver/test_amgpcg.cpp)

# ── 3D standalone solver tests ──
add_executable(test_jacobi_3d solver/test_jacobi_3d.cpp)
target_link_libraries(test_jacobi_3d PRIVATE lfm_lib)

add_executable(test_cg_3d solver/test_cg_3d.cpp)
target_link_libraries(test_cg_3d PRIVATE lfm_lib)

# ── Integration tests (link against lfm_lib) ──
add_executable(test_lfm integration/test_lfm.cpp)
target_link_libraries(test_lfm PRIVATE lfm_lib)

add_executable(test_integration integration/test_integration.cpp)
target_link_libraries(test_integration PRIVATE lfm_lib)

add_executable(test_karman_bench integration/test_karman_bench.cpp)
target_link_libraries(test_karman_bench PRIVATE lfm_lib)

add_executable(test_uaamg integration/test_uaamg.cpp)
target_link_libraries(test_uaamg PRIVATE lfm_lib)

# ── 3D integration tests ──
add_executable(test_uaamg_3d integration/test_uaamg_3d.cpp)
target_link_libraries(test_uaamg_3d PRIVATE lfm_lib)

add_executable(test_karman_bench_3d integration/test_karman_bench_3d.cpp)
target_link_libraries(test_karman_bench_3d PRIVATE lfm_lib)

add_executable(test_karman_validate integration/test_karman_validate.cpp)
target_link_libraries(test_karman_validate PRIVATE lfm_lib)

add_executable(test_solver_perf_3d integration/test_solver_perf_3d.cpp)
target_link_libraries(test_solver_perf_3d PRIVATE lfm_lib)

# ── LFM instability diagnostics ──
add_executable(test_lfm_instability integration/test_lfm_instability.cpp)
target_link_libraries(test_lfm_instability PRIVATE lfm_lib)

add_executable(test_lfm_perturbed_cylinder integration/test_lfm_perturbed_cylinder.cpp)
target_link_libraries(test_lfm_perturbed_cylinder PRIVATE lfm_lib)

# ── Debug tests ──
add_executable(test_pcg_debug debug/test_pcg_debug.cpp)
target_link_libraries(test_pcg_debug PRIVATE lfm_lib)

# Long Karman simulation runner (for animation generation)
add_executable(run_karman_long ../tools/run_karman_long.cpp)
target_link_libraries(run_karman_long PRIVATE lfm_lib)

add_executable(run_karman_chorin ../tools/run_karman_chorin.cpp)
target_link_libraries(run_karman_chorin PRIVATE lfm_lib)

# ── 3D paper scenarios ────────────────────────────────────────────
add_executable(run_vortex_ring_collision   ../tools/run_vortex_ring_collision.cpp)
add_executable(run_vortex_ring_connection  ../tools/run_vortex_ring_connection.cpp)
add_executable(run_trefoil_knot            ../tools/run_trefoil_knot.cpp)
add_executable(run_delta_wing              ../tools/run_delta_wing.cpp)
add_executable(run_fire_ball               ../tools/run_fire_ball.cpp)
add_executable(run_wind_turbine            ../tools/run_wind_turbine.cpp)
foreach(_t run_vortex_ring_collision run_vortex_ring_connection
           run_trefoil_knot run_delta_wing run_fire_ball run_wind_turbine)
    target_link_libraries(${_t} PRIVATE lfm_lib)
endforeach()

# GPU Karman benchmark is defined in top-level CMakeLists.txt (after CUDA setup)
