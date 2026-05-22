# Jacobi iteration (simplest Ax=b solver)
add_executable(jacobi jacobi.cpp)

# RBGS Red-Black Gauss-Seidel smoother
add_executable(rbgs rbgs.cpp)

# CG Conjugate Gradient method
add_executable(cg cg.cpp)

# V-Cycle Multigrid
add_executable(vcycle vcycle.cpp)

# AMGPCG Preconditioned Conjugate Gradient (core paper algorithm)
add_executable(amgpcg amgpcg.cpp)
