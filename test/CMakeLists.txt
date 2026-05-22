cmake_minimum_required(VERSION 3.10)

# Individual unit test executables
add_executable(test_jacobi test_jacobi.cpp)
target_compile_features(test_jacobi PRIVATE cxx_std_17)

add_executable(test_cg test_cg.cpp)
target_compile_features(test_cg PRIVATE cxx_std_17)

add_executable(test_vcycle test_vcycle.cpp)
target_compile_features(test_vcycle PRIVATE cxx_std_17)

add_executable(test_amgpcg test_amgpcg.cpp)
target_compile_features(test_amgpcg PRIVATE cxx_std_17)

add_executable(test_lfm test_lfm.cpp)
target_compile_features(test_lfm PRIVATE cxx_std_17)
