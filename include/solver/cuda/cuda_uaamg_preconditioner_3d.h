#pragma once
#include "solver/cuda/cuda_common_3d.h"
#include <vector>

/// CUDA 3D UAAMG V-cycle preconditioner — matrix-free Galerkin, templated on the
/// scalar type T (double or float). FP32 halves global-memory traffic.
///
/// Call setupLevels(g) once per solve (computes solid hierarchy, Galerkin
/// coefficients, and §5.4 trimming flags), then vcycle_apply(g,r,z) per PCG
/// iteration. apply(g,r,z) does both (standalone use).
template<typename T>
class CudaUAAMGPreconditioner3DT {
public:
    ~CudaUAAMGPreconditioner3DT() { destroy(); }
    void build(const CudaGrid3DT_<T>& fine);
    void setupLevels(const CudaGrid3DT_<T>& fine);
    void vcycle_apply(const CudaGrid3DT_<T>& fine, const T* r, T* z);
    void apply(const CudaGrid3DT_<T>& fine, const T* r, T* z);
    void destroy();

    /// Matrix-free Galerkin stencil per level + §5.4 trimming metadata.
    struct Level {
        CudaGrid3DT_<T> g; int stride = 0;
        T *diag=nullptr, *cx=nullptr, *cy=nullptr, *cz=nullptr;   // cx = +x coupling, diag = row sum
        T cxd=0, cyd=0, czd=0, diagd=0;                          // per-level uniform default stencil
        bool *trimmed=nullptr; int ntx=0, nty=0, ntz=0;          // per-tile "uniform" flag
    };
private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;
};

using CudaUAAMGPreconditioner3D  = CudaUAAMGPreconditioner3DT<double>;
using CudaUAAMGPreconditioner3Dt = CudaUAAMGPreconditioner3DT<float>;   // FP32 Galerkin

/// Legacy float path (non-Galerkin aggregated kernels) — used only by the old
/// test_paper_bench_f benchmark. Superseded by CudaUAAMGPreconditioner3Dt.
class CudaUAAMGPreconditioner3Df {
public:
    ~CudaUAAMGPreconditioner3Df() { destroy(); }
    void build(const CudaGrid3Df& fine);
    void apply_optimized(const CudaGrid3Df& fine, const float* r, float* z);
    void vcycle_only();
    void destroy();
    struct Level { CudaGrid3Df g; int stride = 0; bool* d_trimmed = nullptr; };
private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;
};
