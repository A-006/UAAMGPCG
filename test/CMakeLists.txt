# Add test/ and include/ to include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ── Standalone solver tests (use test_common.h, no lfm_lib needed) ──
add_executable(test_jacobi solver/test_jacobi.cpp)
add_executable(test_cg solver/test_cg.cpp)
add_executable(test_vcycle solver/test_vcycle.cpp)
add_executable(test_amgpcg solver/test_amgpcg.cpp)

# ── Integration tests (link against lfm_lib) ──
add_executable(test_lfm integration/test_lfm.cpp)
target_link_libraries(test_lfm PRIVATE lfm_lib)

add_executable(test_integration integration/test_integration.cpp)
target_link_libraries(test_integration PRIVATE lfm_lib)

add_executable(test_karman_bench integration/test_karman_bench.cpp)
target_link_libraries(test_karman_bench PRIVATE lfm_lib)

add_executable(test_uaamg integration/test_uaamg.cpp)
target_link_libraries(test_uaamg PRIVATE lfm_lib)

# ── Debug tests ──
add_executable(test_pcg_debug debug/test_pcg_debug.cpp)
target_link_libraries(test_pcg_debug PRIVATE lfm_lib)
