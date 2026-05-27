# CUDA UAAMGPCG library (2D)

# CUDA UAAMG library (2D)
add_library(cuda_uaamg_lib STATIC
    cuda_uaamg_preconditioner.cu
    cuda_pcg.cu
    cuda_pcg_solver.cu
)

# CUDA CG library (2D, no preconditioner)
add_library(cuda_cg_lib STATIC
    cuda_cg.cu
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
    cuda_uaamg_preconditioner_3d_opt.cu
    cuda_uaamg_preconditioner_3d_opt_f.cu
    cuda_pcg_3d.cu
    cuda_pcg_3d_opt.cu
    cuda_pcg_solver_3d.cu
)
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

# ── 2D tests ──

add_executable(test_cuda_uaamg test_cuda_uaamg.cu)
set_target_properties(test_cuda_uaamg PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_cuda_uaamg PRIVATE cuda_uaamg_lib lfm_lib)
target_include_directories(test_cuda_uaamg PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_cuda_kernels test_cuda_kernels.cu)
set_target_properties(test_cuda_kernels PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(test_cuda_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_cuda_full test_cuda_full.cu)
set_target_properties(test_cuda_full PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(test_cuda_full PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_apply_twice test_apply_twice.cu)
set_target_properties(test_apply_twice PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_apply_twice PRIVATE cuda_uaamg_lib lfm_lib)
target_include_directories(test_apply_twice PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_cuda_cg test_cuda_cg.cu)
set_target_properties(test_cuda_cg PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_cuda_cg PRIVATE cuda_cg_lib lfm_lib)
target_include_directories(test_cuda_cg PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# ── 3D CUDA tests ──

add_executable(test_cuda_cg_3d test_cuda_cg_3d.cu)
set_target_properties(test_cuda_cg_3d PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_cuda_cg_3d PRIVATE cuda_cg_3d_lib lfm_lib)
target_include_directories(test_cuda_cg_3d PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_cuda_uaamg_3d test_cuda_uaamg_3d.cu)
set_target_properties(test_cuda_uaamg_3d PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_cuda_uaamg_3d PRIVATE cuda_uaamg_3d_lib lfm_lib)
target_include_directories(test_cuda_uaamg_3d PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_solver_perf_3d_gpu test_solver_perf_3d_gpu.cu)
set_target_properties(test_solver_perf_3d_gpu PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_solver_perf_3d_gpu PRIVATE cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
target_include_directories(test_solver_perf_3d_gpu PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_paper_bench test_paper_bench.cu)
set_target_properties(test_paper_bench PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_paper_bench PRIVATE cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
target_include_directories(test_paper_bench PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_paper_bench_f test_paper_bench_f.cu)
set_target_properties(test_paper_bench_f PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_paper_bench_f PRIVATE cuda_uaamg_3d_lib cuda_cg_3d_lib lfm_lib)
target_include_directories(test_paper_bench_f PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
