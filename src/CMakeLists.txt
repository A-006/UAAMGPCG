cmake_minimum_required(VERSION 3.10)

# LFM 2D fluid simulation (Jacobi solves the pressure Poisson equation)
add_executable(lfm_2d lfm_2d.cpp)
target_compile_features(lfm_2d PRIVATE cxx_std_17)
target_compile_options(lfm_2d PRIVATE -O2)
