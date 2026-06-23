# ── CUDA GPU unit / benchmark tests ──
# Moved here from src/solver/cuda. They link the cuda_*_lib targets built in
# src/solver/cuda, so this directory is add_subdirectory'd from the top-level
# CMakeLists CUDA block *after* those libraries are defined.

function(add_cuda_test name)
    add_executable(${name} ${name}.cu)
    set_target_properties(${name} PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if(ARGN)
        target_link_libraries(${name} PRIVATE ${ARGN})
    endif()
endfunction()

# ── Semi-structured matrix-free MG on GPU (header-only; arXiv:2604.18886) ──
add_cuda_test(test_semistruct_gpu_2d)

# ── 2D ──
add_cuda_test(test_cuda_uaamg    cuda_uaamg_lib lfm_lib)
add_cuda_test(test_cuda_kernels)
add_cuda_test(test_cuda_full)
add_cuda_test(test_apply_twice   cuda_uaamg_lib lfm_lib)
add_cuda_test(test_cuda_cg       cuda_cg_lib lfm_lib)

# ── 3D ──
add_cuda_test(test_cuda_cg_3d         cuda_cg_3d_lib lfm_lib)
add_cuda_test(test_cuda_uaamg_3d      cuda_uaamg_3d_lib lfm_lib)
add_cuda_test(test_solver_perf_3d_gpu cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
add_cuda_test(test_paper_bench        cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
add_cuda_test(test_paper_bench_f      cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
add_cuda_test(test_fp32_vcycle        cuda_uaamg_3d_lib)
add_cuda_test(bench_vcycle256         cuda_uaamg_3d_lib)
add_cuda_test(bench_vcycle_fresh      cuda_uaamg_3d_lib)
add_cuda_test(conv2                  cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
add_cuda_test(test_tiled_soa)
add_cuda_test(test_mixed_solve        cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
add_cuda_test(test_cuda_lfm_3d        cuda_lfm_3d_lib cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
add_cuda_test(bench_solve             cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
add_cuda_test(solid_check            cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
