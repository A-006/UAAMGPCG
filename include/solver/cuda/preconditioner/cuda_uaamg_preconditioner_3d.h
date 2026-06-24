#pragma once
#include "solver/cuda/cuda_common_3d.h"
#include <vector>

/// CUDA 3D UAAMG V-cycle preconditioner — matrix-free Galerkin, templated on the
/// scalar type T (double or float). FP32 halves global-memory traffic.
///
/// Call setupLevels(g) once per solve (computes solid hierarchy, Galerkin
/// coefficients, and §5.4 trimming flags), then vcycle_apply(g,r,z) per PCG
/// iteration. apply(g,r,z) does both (standalone use).
template <typename T> class CudaUAAMGPreconditioner3DT {
public:
    ~CudaUAAMGPreconditioner3DT() {
        destroy();
    }
    void build(const CudaGrid3DT_<T>& fine);
    void setupLevels(const CudaGrid3DT_<T>& fine);
    // Force the next setupLevels() to rebuild the Galerkin/trimming coefficients even
    // if the geometry signature matches the cache. Call this whenever the solid mask is
    // mutated in place (moving geometry); static-geometry scenarios never need it.
    void invalidate_coeffs() { coeffs_ready_ = false; }
    void vcycle_apply(const CudaGrid3DT_<T>& fine, const T* r, T* z);
    void apply(const CudaGrid3DT_<T>& fine, const T* r, T* z);
    void destroy();

    // ── Tile-native fast path (PCG stays in the tile layout; no per-iter convert) ──
    long level0_count() const {
        return levels_.empty() ? 0 : (long)levels_[0].g.num_tiles * 512;
    }
    const bool* level0_solid() const {
        return levels_.empty() ? nullptr : levels_[0].g.solid;
    }
    // The finest-level tile b/x ARE the PCG's r and z (aliased) — no copy needed:
    // the V-cycle reads b (=r) and writes x (=z) in place, so the entire solve runs
    // in the tile layout with zero per-iteration pitched↔tile conversion.
    T* level0_b() {
        return levels_.empty() ? nullptr : levels_[0].g.b;
    }
    T* level0_x() {
        return levels_.empty() ? nullptr : levels_[0].g.x;
    }
    void to_tile(const T* pitched, T* tile, const CudaGrid3DT_<T>& fine);   // scatter (once)
    void from_tile(const T* tile, T* pitched, const CudaGrid3DT_<T>& fine); // gather (once)
    void matvec_tiled(const T* p_tile, T* Ap_tile);                        // A·p in tile layout
    void vcycle_inplace(); // M⁻¹ in place: b(=r) already set → result in x(=z)
    // Same as vcycle_inplace but WITHOUT the terminal cudaDeviceSynchronize: the
    // V-cycle's kernels are issued stream-ordered on stream 0, so a device-resident
    // PCG whose other ops also run on stream 0 sees them in order with no host sync.
    // (The host-side x↔scratch ping-pong still happens, so callers must re-fetch
    //  level0_x() afterwards, exactly as for vcycle_inplace.)
    void vcycle_inplace_async();

    /// Matrix-free Galerkin stencil per level + §5.4 trimming metadata.
    struct Level {
        CudaGrid3DT_<T> g;
        int stride = 0;
        T *diag = nullptr, *cx = nullptr, *cy = nullptr,
          *cz = nullptr;                        // cx = +x coupling, diag = row sum
        T cxd = 0, cyd = 0, czd = 0, diagd = 0; // per-level uniform default stencil
        bool* trimmed = nullptr;
        int ntx = 0, nty = 0, ntz = 0; // per-tile "uniform" flag
        T* scratch    = nullptr;       // ping-pong buffer for the fused post-smooth
    };

private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;
    // Coefficient-hierarchy cache (lever #1). The solid restriction, fine/Galerkin
    // coeffs and §5.4 trimming computed in setupLevels() depend ONLY on the solid mask
    // and grid spacing — not the RHS — and the coeff buffers are read-only during the
    // solve. So once built they can be reused across every projection of a static-
    // geometry run (all current LFM scenarios). The guard keys on the solid-mask pointer
    // + spacing; invalidate_coeffs() (or env UAAMG_REBUILD=1) forces a rebuild.
    const void* coeff_sig_solid_ = nullptr;
    T coeff_sig_dx_ = 0, coeff_sig_dy_ = 0, coeff_sig_dz_ = 0;
    bool coeffs_ready_ = false;
};

using CudaUAAMGPreconditioner3D  = CudaUAAMGPreconditioner3DT<double>;
using CudaUAAMGPreconditioner3Dt = CudaUAAMGPreconditioner3DT<float>; // FP32 Galerkin

/// Legacy float path (non-Galerkin aggregated kernels) — used only by the old
/// test_paper_bench_f benchmark. Superseded by CudaUAAMGPreconditioner3Dt.
class CudaUAAMGPreconditioner3Df {
public:
    ~CudaUAAMGPreconditioner3Df() {
        destroy();
    }
    void build(const CudaGrid3Df& fine);
    void apply_optimized(const CudaGrid3Df& fine, const float* r, float* z);
    void vcycle_only();
    void destroy();
    struct Level {
        CudaGrid3Df g;
        int stride      = 0;
        bool* d_trimmed = nullptr;
    };

private:
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1, cached_nz_ = -1;
};
