cmake_minimum_required(VERSION 3.10)

# Jacobi iteration (simplest Ax=b solver)
add_executable(jacobi jacobi.cpp)
target_compile_features(jacobi PRIVATE cxx_std_17)

# RBGS Red-Black Gauss-Seidel smoother
add_executable(rbgs rbgs.cpp)
target_compile_features(rbgs PRIVATE cxx_std_17)

# CG Conjugate Gradient method
add_executable(cg cg.cpp)
target_compile_features(cg PRIVATE cxx_std_17)

# V-Cycle Multigrid
add_executable(vcycle vcycle.cpp)
target_compile_features(vcycle PRIVATE cxx_std_17)

# AMGPCG Preconditioned Conjugate Gradient (core paper algorithm)
add_executable(amgpcg amgpcg.cpp)
target_compile_features(amgpcg PRIVATE cxx_std_17)
