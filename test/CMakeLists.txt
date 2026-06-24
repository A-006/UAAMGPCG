# Add test/ and include/ to include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ── Semi-structured matrix-free multigrid (header-only, no lfm_lib) ──
# Reproduction of Wang/Sun/Zhu arXiv:2604.18886 (see include/semistruct/).
foreach(_sst test_operator_2d test_laplacian_accuracy_2d test_poisson_convergence_2d
             test_operator_3d test_laplacian_accuracy_3d test_poisson_convergence_3d
             test_cutcell_2d test_cutcell_3d)
    add_executable(${_sst} semistruct/${_sst}.cpp)
endforeach()

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

add_executable(test_bc_solid integration/test_bc_solid.cpp)
target_link_libraries(test_bc_solid PRIVATE lfm_lib)

# ── Bulk CPU unit tests (added 2026-06-17) — all link lfm_lib only ──
foreach(_ut
        test_bc_patches_2d test_grid_ops_2d test_grid_3d
        test_bc_patches_3d test_config_cli test_scenarios_2d test_scenarios_3d
        test_io_roundtrip test_numerics_interp test_flow_map)
    add_executable(${_ut} integration/${_ut}.cpp)
    target_link_libraries(${_ut} PRIVATE lfm_lib)
endforeach()

add_executable(test_karman_bench integration/test_karman_bench.cpp)
target_link_libraries(test_karman_bench PRIVATE lfm_lib)

add_executable(test_uaamg integration/test_uaamg.cpp)
target_link_libraries(test_uaamg PRIVATE lfm_lib)

# ── 3D integration tests ──
add_executable(test_uaamg_3d integration/test_uaamg_3d.cpp)
target_link_libraries(test_uaamg_3d PRIVATE lfm_lib)

add_executable(test_lfm_3d integration/test_lfm_3d.cpp)
target_link_libraries(test_lfm_3d PRIVATE lfm_lib)

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

# Long Cylinder simulation runner (for animation generation)
add_executable(run_karman_long ../tools/run_karman_long.cpp)
target_link_libraries(run_karman_long PRIVATE lfm_lib)

add_executable(run_karman_chorin ../tools/run_karman_chorin.cpp)
target_link_libraries(run_karman_chorin PRIVATE lfm_lib)

# ── 3D paper scenarios ────────────────────────────────────────────
# Note: vortex reconnection (Fig. 5) and trefoil knot (Fig. 7) are now scenarios
# of the unified `cfdsim` launcher (scenario=vortex_reconnection / trefoil_knot).
add_executable(run_vortex_ring_collision   ../tools/run_vortex_ring_collision.cpp)
add_executable(run_delta_wing              ../tools/run_delta_wing.cpp)
add_executable(test_bfecc_clamp            integration/test_bfecc_clamp.cpp)
add_executable(run_delta_wing_lfm          ../tools/run_delta_wing_lfm.cpp)
add_executable(dump_collision_ic           ../tools/dump_collision_ic.cpp)
foreach(_t run_vortex_ring_collision run_delta_wing
           test_bfecc_clamp run_delta_wing_lfm dump_collision_ic)
    target_link_libraries(${_t} PRIVATE lfm_lib)
endforeach()

# GPU Cylinder benchmark is defined in top-level CMakeLists.txt (after CUDA setup)
