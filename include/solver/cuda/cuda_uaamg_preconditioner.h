#pragma once
#include "solver/cuda/cuda_common.h"
#include <vector>

/// CUDA UAAMG V-cycle preconditioner — Section 5.1, Algorithm 3.
///
/// - SOA data layout with 8x8 tiling
/// - Aggregated RBGS + restriction kernel (Section 5.3)
/// - Scale-2 prolongation (Stuben 2001)
/// - Matrix-free Galerkin coarse operators
class CudaUAAMGPreconditioner {
public:
    ~CudaUAAMGPreconditioner() { destroy(); }

    /// Build multi-level hierarchy on the GPU.
    void build(const CudaGrid& fine);

    /// Apply one V-cycle: z = M^{-1} * r  (on device).
    void apply(const CudaGrid& fine, const double* r, double* z);

    void destroy();

    struct Level {
        CudaGrid g;
        int stride = 0;
    };

private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1;
};
