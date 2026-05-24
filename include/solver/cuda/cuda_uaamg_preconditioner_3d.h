#pragma once
#include "solver/cuda/cuda_common_3d.h"
#include <vector>

/// CUDA 3D UAAMG V-cycle preconditioner (double).
class CudaUAAMGPreconditioner3D {
public:
    ~CudaUAAMGPreconditioner3D() { destroy(); }
    void build(const CudaGrid3D& fine);
    void apply(const CudaGrid3D& fine, const double* r, double* z);
    void apply_optimized(const CudaGrid3D& fine, const double* r, double* z);
    void destroy();
    struct Level { CudaGrid3D g; int stride = 0; };
private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;
};

/// CUDA 3D UAAMG V-cycle preconditioner (float) — with coefficient trimming.
class CudaUAAMGPreconditioner3Df {
public:
    ~CudaUAAMGPreconditioner3Df() { destroy(); }
    void build(const CudaGrid3Df& fine);
    void apply_optimized(const CudaGrid3Df& fine, const float* r, float* z);
    void destroy();
    struct Level { CudaGrid3Df g; int stride = 0; bool* d_trimmed = nullptr; };
private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;
};
