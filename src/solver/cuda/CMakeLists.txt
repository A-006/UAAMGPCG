# CUDA UAAMGPCG library

# CUDA UAAMG library
add_library(cuda_uaamg_lib STATIC
    cuda_uaamg_preconditioner.cu
    cuda_pcg.cu
)

set_target_properties(cuda_uaamg_lib PROPERTIES
    LINKER_LANGUAGE CUDA
)

target_include_directories(cuda_uaamg_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA solver test
add_executable(test_cuda_uaamg test_cuda_uaamg.cu)
set_target_properties(test_cuda_uaamg PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(test_cuda_uaamg PRIVATE cuda_uaamg_lib lfm_lib)
target_include_directories(test_cuda_uaamg PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# Micro RBGS test
add_executable(test_rbgs_micro test_rbgs_micro.cu)
set_target_properties(test_rbgs_micro PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(test_rbgs_micro PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# Per-kernel unit tests
add_executable(test_cuda_kernels test_cuda_kernels.cu)
set_target_properties(test_cuda_kernels PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(test_cuda_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# Full GPU tests (every kernel + PCG trace)
add_executable(test_cuda_full test_cuda_full.cu)
set_target_properties(test_cuda_full PROPERTIES LINKER_LANGUAGE CUDA)
target_include_directories(test_cuda_full PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
