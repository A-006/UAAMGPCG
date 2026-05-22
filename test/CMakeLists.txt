# Standalone solver tests (self-contained, use test_common.h)
add_executable(test_jacobi test_jacobi.cpp)
add_executable(test_cg test_cg.cpp)
add_executable(test_vcycle test_vcycle.cpp)
add_executable(test_amgpcg test_amgpcg.cpp)

# LFM tests link against lfm_lib
add_executable(test_lfm test_lfm.cpp)
target_link_libraries(test_lfm PRIVATE lfm_lib)

add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration PRIVATE lfm_lib)
