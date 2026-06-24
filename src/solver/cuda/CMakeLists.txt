# CUDA UAAMGPCG library (2D)

# CUDA UAAMG library (2D)
add_library(cuda_uaamg_lib STATIC
    cuda_uaamg_preconditioner_2d.cu
    cuda_pcg_2d.cu
    cuda_pcg_solver_2d.cu
)

# CUDA CG library (2D, no preconditioner)
add_library(cuda_cg_lib STATIC
    cuda_cg_2d.cu
)
set_target_properties(cuda_cg_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_cg_lib PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(cuda_uaamg_lib PROPERTIES
    LINKER_LANGUAGE CUDA
)

target_include_directories(cuda_uaamg_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ── 3D CUDA libraries ──

add_library(cuda_uaamg_3d_lib STATIC
    cuda_uaamg_preconditioner_3d.cu
    cuda_uaamg_preconditioner_3d_opt_f.cu
    cuda_pcg_3d.cu
    cuda_pcg_3d_opt.cu
    cuda_pcg_solver_3d.cu
)
# cuda_uaamg_preconditioner_3d_opt.cu (double aggregated V-cycle) dropped:
# break-even in FP64 + nx≥256 bug; production uses the templated separate path.
set_target_properties(cuda_uaamg_3d_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_uaamg_3d_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(cuda_cg_3d_lib STATIC
    cuda_cg_3d.cu
)
set_target_properties(cuda_cg_3d_lib PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(cuda_cg_3d_lib PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# ── 3D LFM GPU simulator (device-resident full LFM cycle) ──
add_library(cuda_lfm_3d_lib STATIC
    cuda_lfm_3d_kernels.cu
    cuda_lfm_simulator_3d.cu
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
