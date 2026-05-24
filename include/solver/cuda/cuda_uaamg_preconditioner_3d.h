#pragma once
#include "solver/cuda/cuda_common_3d.h"
#include <vector>

/// CUDA 3D UAAMG V-cycle preconditioner.
///
/// - 7-point stencil (3D 7-point Laplacian)
/// - Aggregated RBGS + restriction kernel (8-to-1, 2x2x2)
/// - Scale-2 prolongation (Stuben 2001)
/// - Matrix-free Galerkin coarse operators
class CudaUAAMGPreconditioner3D {
public:
    ~CudaUAAMGPreconditioner3D() { destroy(); }

    /// Build multi-level hierarchy on the GPU.
    void build(const CudaGrid3D& fine);

    /// Apply one V-cycle: z = M^{-1} * r  (on device).
    void apply(const CudaGrid3D& fine, const double* r, double* z);

    /// Optimized V-cycle (aggregated kernels, shared-memory tiling).
    void apply_optimized(const CudaGrid3D& fine, const double* r, double* z);

    void destroy();

    struct Level {
        CudaGrid3D g;
        int stride = 0;
    };

private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;
};
