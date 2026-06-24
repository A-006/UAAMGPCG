# CUDA solver libraries. Sources are organized like the CPU side:
#   krylov/         — cuda_cg, cuda_pcg, cuda_pcg_solver
#   preconditioner/ — cuda_uaamg_preconditioner (+ opt variants)
# cuda_common_*.h and prof_vcycle.cu stay at this directory root.

# ── 2D ──
add_library(cuda_uaamg_lib STATIC
    preconditioner/cuda_uaamg_preconditioner_2d.cu
    krylov/cuda_pcg_2d.cu
    krylov/cuda_pcg_solver_2d.cu
)
set_target_properties(cuda_uaamg_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_uaamg_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(cuda_cg_lib STATIC
    krylov/cuda_cg_2d.cu
)
set_target_properties(cuda_cg_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_cg_lib PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# ── 3D ──
add_library(cuda_uaamg_3d_lib STATIC
    preconditioner/cuda_uaamg_preconditioner_3d.cu
    preconditioner/cuda_uaamg_preconditioner_3d_opt_f.cu
    krylov/cuda_pcg_3d.cu
    krylov/cuda_pcg_3d_opt.cu
    krylov/cuda_pcg_solver_3d.cu
)
# cuda_uaamg_preconditioner_3d_opt.cu (double aggregated V-cycle) dropped:
# break-even in FP64 + nx≥256 bug; production uses the templated separate path.
set_target_properties(cuda_uaamg_3d_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_uaamg_3d_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(cuda_cg_3d_lib STATIC
    krylov/cuda_cg_3d.cu
)
set_target_properties(cuda_cg_3d_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_cg_3d_lib PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# ── 3D LFM GPU simulator (device-resident full LFM cycle) ──
# It is a time integrator, not a linear solver, so its sources live under
# integrator/lfm/; the build target stays here to sit next to cuda_uaamg_3d_lib.
add_library(cuda_lfm_3d_lib STATIC
    ${CMAKE_SOURCE_DIR}/src/integrator/lfm/cuda_lfm_3d_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/integrator/lfm/cuda_lfm_simulator_3d.cu
)
set_target_properties(cuda_lfm_3d_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_lfm_3d_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Disable fused multiply-add contraction so the device arithmetic matches the
# CPU LFMSimulator3D reference bit-closely. The flow-map RK4 marching is
# ill-conditioned in vortex cores, where FMA-vs-(mul+add) rounding differences
# amplify into visible local discrepancies; without this the GPU cycle drifts
# from the CPU golden reference by ~1e-2 in a few cells (energy still matches).
target_compile_options(cuda_lfm_3d_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
target_link_libraries(cuda_lfm_3d_lib PUBLIC cuda_uaamg_3d_lib)

# ── Profiling tool (not a unit test; GPU unit tests live in test/cuda) ──
add_executable(prof_vcycle prof_vcycle.cu)
set_target_properties(prof_vcycle PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(prof_vcycle PRIVATE cuda_uaamg_3d_lib)
target_include_directories(prof_vcycle PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
