// ════════════════════════════════════════════════════════════════════
// GPU 3D LFM kernels — device port of src/simulator/lfm_simulator_3d.cpp.
// This translation unit holds the device-resident state management plus the
// per-stage kernels. Kernels are added phase-by-phase (see plan); each is
// validated against the CPU LFMSimulator3D golden reference in
// test/cuda/test_cuda_lfm_3d.cu.
//
// P0: state alloc/free + H2D/D2H + free-slip box BC + Poisson projection.
// ════════════════════════════════════════════════════════════════════
#include "solver/cuda/cuda_lfm_3d.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <vector>

namespace {
inline __half* hmalloc(int n) {
    __half* p = nullptr;
    cudaMalloc(&p, (size_t)n * sizeof(__half));
    cudaMemset(p, 0, (size_t)n * sizeof(__half));
    return p;
}
inline void hfree(void*& p) {
    if (p)
        cudaFree(p);
    p = nullptr;
}
inline double* dmalloc(int n) {
    double* p = nullptr;
    cudaMalloc(&p, (size_t)n * sizeof(double));
    cudaMemset(p, 0, (size_t)n * sizeof(double));
    return p;
}
inline void dfree(double*& p) {
    if (p)
        cudaFree(p);
    p = nullptr;
}
inline float* fmalloc(int n) {
    float* p = nullptr;
    cudaMalloc(&p, (size_t)n * sizeof(float));
    cudaMemset(p, 0, (size_t)n * sizeof(float));
    return p;
}
inline void ffree(float*& p) {
    if (p)
        cudaFree(p);
    p = nullptr;
}
dim3 block3() { return dim3(8, 8, 8); }
dim3 grid3(int nx, int ny, int nz) {
    return dim3((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
}
} // namespace

// Element-wise double→float copy of one MAC component (coalesced 1D map). Used to
// stage the FP32 velocity/source scratch so the hot sampling gather is pure FP32.
__global__ void d2f_copy_kernel(float* __restrict dst, const double* __restrict src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = (float)src[i];
}
static void convert_vel_f32(CudaVel3D src, float* du, float* dv, float* dw, int us, int vs,
                            int ws) {
    int B = 256;
    d2f_copy_kernel<<<(us + B - 1) / B, B>>>(du, src.u, us);
    d2f_copy_kernel<<<(vs + B - 1) / B, B>>>(dv, src.v, vs);
    d2f_copy_kernel<<<(ws + B - 1) / B, B>>>(dw, src.w, ws);
}

// Element-wise double→__half copy of one MAC component. Stages the FP16 sampling
// scratch: the hot gather then loads 2 bytes/sample (vs 4 for FP32) and the inner
// B-spline runs packed half2. |u|~5 is well inside FP16's range, only precision
// (10-bit mantissa) is lost — that's the experiment's whole tradeoff.
__global__ void d2h_copy_kernel(__half* __restrict dst, const double* __restrict src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __double2half(src[i]);
}
static void convert_vel_f16(CudaVel3D src, __half* du, __half* dv, __half* dw, int us, int vs,
                            int ws) {
    int B = 256;
    d2h_copy_kernel<<<(us + B - 1) / B, B>>>(du, src.u, us);
    d2h_copy_kernel<<<(vs + B - 1) / B, B>>>(dv, src.v, vs);
    d2h_copy_kernel<<<(ws + B - 1) / B, B>>>(dw, src.w, ws);
}

// Resolve the FP16-sampling switch: the state flag (cfg.lfm_sample_fp16) OR env
// LFM_FP16=1. Env is cached once. Only meaningful when fp32_march is also set.
static bool fp16_sampling(const CudaLFMState3D& s) {
    static const bool env = std::getenv("LFM_FP16") != nullptr;
    return s.fp32_march && (s.sample_fp16 || env);
}

// ════════════════════════════════════════════════════════════════════
// Shared device functions — exact ports of the corresponding routines in
// src/simulator/lfm_simulator_3d.cpp (same clamps, B-spline weights, MAC
// face offsets, and 27-point summation order, so GPU == CPU to fp rounding).
// ════════════════════════════════════════════════════════════════════
__device__ inline double dclamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__device__ inline int iclamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__device__ inline void d_bspline(double r, double w[3]) {
    double a = 0.5 - r, b = 0.5 + r;
    w[0]     = 0.5 * a * a;
    w[1]     = 0.75 - r * r;
    w[2]     = 0.5 * b * b;
}
// d/dr of the quadratic B-spline weights (the analytic spline derivative, ≡ author's dN2).
// Used by d_sample_velocity_grad so ∇u is the exact derivative of the SAME interpolant
// that produces u — matching InterpMacN2Grad instead of a nearest-cell finite difference.
__device__ inline void d_bspline_d(double r, double dw[3]) {
    dw[0] = r - 0.5;       // d/dr [0.5(0.5-r)^2]
    dw[1] = -2.0 * r;      // d/dr [0.75 - r^2]
    dw[2] = r + 0.5;       // d/dr [0.5(0.5+r)^2]
}

// ── T-typed helpers (T=double → bit-exact path; T=float → fast production
// LFM sampling). Velocity/map loads stay from the double arrays and are cast to
// T; only the per-point ALU (the hot 27-point interpolation) runs in T. ──
template <typename T> __device__ inline T tclamp(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
template <typename T> __device__ inline void t_bspline(T r, T w[3]) {
    T a  = T(0.5) - r, b = T(0.5) + r;
    w[0] = T(0.5) * a * a;
    w[1] = T(0.75) - r * r;
    w[2] = T(0.5) * b * b;
}
template <typename T> __device__ inline void t_bspline_d(T r, T dw[3]) {
    dw[0] = r - T(0.5);
    dw[1] = T(-2.0) * r;
    dw[2] = r + T(0.5);
}
// Nearest-cell index of a continuous grid coordinate. The FP64 path keeps the
// EXACT expression of the CPU reference ((int)floor((double)c + 0.5)) so the
// bit-exact GPU-vs-CPU test still passes; the FP32 path uses floorf so the hot
// production sampler stays off the RTX 3090's 1/64-rate FP64 pipe (ncu showed
// the FP64 pipe was the #1 hotspot at ~24%, all from this floor((double)…)).
template <typename T> __device__ inline int t_round_idx(T c);
template <> __device__ inline int t_round_idx<double>(double c) {
    return (int)floor(c + 0.5);
}
template <> __device__ inline int t_round_idx<float>(float c) {
    return (int)floorf(c + 0.5f);
}

// 27-point quadratic B-spline velocity interpolation (MAC-aware). VT is the
// stored velocity element type: double (bit-exact path) or float (fp32 path,
// pre-staged scratch → no per-load FP64 convert). Deduced from the pointer args.
template <typename T, typename VT = double>
__device__ inline void d_sample_velocity(const VT* uu, const VT* vv, const VT* ww,
                                         T x, T y, T z, int nx, int ny, int nz,
                                         T dx, T dy, T dz, T Lx, T Ly,
                                         T Lz, T& vu, T& vvel, T& vw) {
    x = tclamp<T>(x, T(0), Lx);
    y = tclamp<T>(y, T(0), Ly);
    z = tclamp<T>(z, T(0), Lz);

    // u-face at (i·dx,(j-0.5)·dy,(k-0.5)·dz)
    {
        T cx = x / dx, cy = y / dy + T(0.5), cz = z / dz + T(0.5);
        int ic = t_round_idx<T>(cx), jc = t_round_idx<T>(cy), kc = t_round_idx<T>(cz);
        T wx[3], wy[3], wz[3];
        t_bspline<T>(cx - ic, wx);
        t_bspline<T>(cy - jc, wy);
        t_bspline<T>(cz - kc, wz);
        T s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 0, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * (T)__ldg(&uu[lfm_iu(ii, jj, kk, nx, ny)]);
                }
        vu = s;
    }
    // v-face at ((i-0.5)·dx, j·dy, (k-0.5)·dz)
    {
        T cx = x / dx + T(0.5), cy = y / dy, cz = z / dz + T(0.5);
        int ic = t_round_idx<T>(cx), jc = t_round_idx<T>(cy), kc = t_round_idx<T>(cz);
        T wx[3], wy[3], wz[3];
        t_bspline<T>(cx - ic, wx);
        t_bspline<T>(cy - jc, wy);
        t_bspline<T>(cz - kc, wz);
        T s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 0, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * (T)__ldg(&vv[lfm_iv(ii, jj, kk, nx, ny)]);
                }
        vvel = s;
    }
    // w-face at ((i-0.5)·dx,(j-0.5)·dy, k·dz)
    {
        T cx = x / dx + T(0.5), cy = y / dy + T(0.5), cz = z / dz;
        int ic = t_round_idx<T>(cx), jc = t_round_idx<T>(cy), kc = t_round_idx<T>(cz);
        T wx[3], wy[3], wz[3];
        t_bspline<T>(cx - ic, wx);
        t_bspline<T>(cy - jc, wy);
        t_bspline<T>(cz - kc, wz);
        T s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 0, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * (T)__ldg(&ww[lfm_iw(ii, jj, kk, nx, ny)]);
                }
        vw = s;
    }
}

// Velocity AND its 3x3 spatial gradient from the SAME 27-point quadratic B-spline as
// d_sample_velocity. g[3a+b] = ∂u_a/∂x_b, the analytic spline derivative at the exact
// off-grid position (≡ author's InterpMacN2Grad). Replaces the nearest-cell finite
// difference so the flow-map Jacobian dF/dt=∇u·F is evolved consistently — this is the
// circulation-preserving term, and the FD/snap version was the dominant dissipation source.
template <typename T, typename VT = double>
__device__ inline void d_sample_velocity_grad(const VT* uu, const VT* vv, const VT* ww,
                                              T x, T y, T z, int nx, int ny, int nz,
                                              T dx, T dy, T dz, T Lx, T Ly,
                                              T Lz, T& vu, T& vvel, T& vw,
                                              T g[9]) {
    x = tclamp<T>(x, T(0), Lx);
    y = tclamp<T>(y, T(0), Ly);
    z = tclamp<T>(z, T(0), Lz);
    // u-face at (i·dx,(j-0.5)·dy,(k-0.5)·dz)  → row a=0
    {
        T cx = x / dx, cy = y / dy + T(0.5), cz = z / dz + T(0.5);
        int ic = t_round_idx<T>(cx), jc = t_round_idx<T>(cy), kc = t_round_idx<T>(cz);
        T wx[3], wy[3], wz[3], dwx[3], dwy[3], dwz[3];
        t_bspline<T>(cx - ic, wx);   t_bspline_d<T>(cx - ic, dwx);
        t_bspline<T>(cy - jc, wy);   t_bspline_d<T>(cy - jc, dwy);
        t_bspline<T>(cz - kc, wz);   t_bspline_d<T>(cz - kc, dwz);
        T s = 0, sx = 0, sy = 0, sz = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 0, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    T val = (T)__ldg(&uu[lfm_iu(ii, jj, kk, nx, ny)]);
                    s  += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * val;
                    sx += dwx[di + 1] * wy[dj + 1] * wz[dk + 1] * val;
                    sy += wx[di + 1] * dwy[dj + 1] * wz[dk + 1] * val;
                    sz += wx[di + 1] * wy[dj + 1] * dwz[dk + 1] * val;
                }
        vu   = s;
        g[0] = sx / dx;  g[1] = sy / dy;  g[2] = sz / dz;
    }
    // v-face at ((i-0.5)·dx, j·dy, (k-0.5)·dz)  → row a=1
    {
        T cx = x / dx + T(0.5), cy = y / dy, cz = z / dz + T(0.5);
        int ic = t_round_idx<T>(cx), jc = t_round_idx<T>(cy), kc = t_round_idx<T>(cz);
        T wx[3], wy[3], wz[3], dwx[3], dwy[3], dwz[3];
        t_bspline<T>(cx - ic, wx);   t_bspline_d<T>(cx - ic, dwx);
        t_bspline<T>(cy - jc, wy);   t_bspline_d<T>(cy - jc, dwy);
        t_bspline<T>(cz - kc, wz);   t_bspline_d<T>(cz - kc, dwz);
        T s = 0, sx = 0, sy = 0, sz = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 0, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    T val = (T)__ldg(&vv[lfm_iv(ii, jj, kk, nx, ny)]);
                    s  += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * val;
                    sx += dwx[di + 1] * wy[dj + 1] * wz[dk + 1] * val;
                    sy += wx[di + 1] * dwy[dj + 1] * wz[dk + 1] * val;
                    sz += wx[di + 1] * wy[dj + 1] * dwz[dk + 1] * val;
                }
        vvel = s;
        g[3] = sx / dx;  g[4] = sy / dy;  g[5] = sz / dz;
    }
    // w-face at ((i-0.5)·dx,(j-0.5)·dy, k·dz)  → row a=2
    {
        T cx = x / dx + T(0.5), cy = y / dy + T(0.5), cz = z / dz;
        int ic = t_round_idx<T>(cx), jc = t_round_idx<T>(cy), kc = t_round_idx<T>(cz);
        T wx[3], wy[3], wz[3], dwx[3], dwy[3], dwz[3];
        t_bspline<T>(cx - ic, wx);   t_bspline_d<T>(cx - ic, dwx);
        t_bspline<T>(cy - jc, wy);   t_bspline_d<T>(cy - jc, dwy);
        t_bspline<T>(cz - kc, wz);   t_bspline_d<T>(cz - kc, dwz);
        T s = 0, sx = 0, sy = 0, sz = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 0, nz);
                    T val = (T)__ldg(&ww[lfm_iw(ii, jj, kk, nx, ny)]);
                    s  += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * val;
                    sx += dwx[di + 1] * wy[dj + 1] * wz[dk + 1] * val;
                    sy += wx[di + 1] * dwy[dj + 1] * wz[dk + 1] * val;
                    sz += wx[di + 1] * wy[dj + 1] * dwz[dk + 1] * val;
                }
        vw   = s;
        g[6] = sx / dx;  g[7] = sy / dy;  g[8] = sz / dz;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SEPARABLE FP32 samplers — the tensor-product quadratic B-spline factored into
// three sequential 1-D reductions (di → dj → dk) instead of the naive 27-tap
// triple product. Same gather (27 values, identical clamps/MAC offsets), but the
// weight algebra collapses: the value sampler drops from ~2 mul/tap to a chain of
// 1-D dot products, and the grad sampler computes (value,∂x,∂y,∂z) from shared
// partials instead of four independent triple products per tap (~324→~90 FMAs).
// FP32-only: the di→dj→dk summation order differs from the bit-exact FP64 nested
// loop, so these are used solely when T=VT=float; the double template keeps the
// original order and stays GPU==CPU. Verified numerically by P2/P3/P4 (FP32
// "matches CPU" checks use a tolerance, not bit-exactness).
// ════════════════════════════════════════════════════════════════════════════

// One MAC face's separable value interpolation (27-tap). lo* are the per-axis
// clamp lower bounds (upper = nx/ny/nz). Returns Σ wx·wy·wz·val.
__device__ inline float d_face_value_f32(const float* __restrict fld, int axis, float cx, float cy,
                                         float cz, int nx, int ny, int nz, int ilo, int jlo,
                                         int klo) {
    int ic = t_round_idx<float>(cx), jc = t_round_idx<float>(cy), kc = t_round_idx<float>(cz);
    float wx[3], wy[3], wz[3];
    t_bspline<float>(cx - ic, wx);
    t_bspline<float>(cy - jc, wy);
    t_bspline<float>(cz - kc, wz);
    int ii0 = iclamp(ic - 1, ilo, nx), ii1 = iclamp(ic, ilo, nx), ii2 = iclamp(ic + 1, ilo, nx);
    // Hoist the per-axis flat strides so the inner gather has no axis branch and
    // reuses a single (k,j) base offset for all three di taps.
    int sj = (axis == 0) ? (nx + 1) : (nx + 2);
    int sk = (axis == 0) ? (nx + 1) * (ny + 2)
                         : (axis == 1 ? (nx + 2) * (ny + 1) : (nx + 2) * (ny + 2));
    float s = 0.f;
    for (int dk = -1; dk <= 1; dk++) {
        int kb = iclamp(kc + dk, klo, nz) * sk;
        float pk = 0.f;
        for (int dj = -1; dj <= 1; dj++) {
            int base = kb + iclamp(jc + dj, jlo, ny) * sj;
            float r = wx[0] * __ldg(&fld[base + ii0]) + wx[1] * __ldg(&fld[base + ii1]) +
                      wx[2] * __ldg(&fld[base + ii2]);
            pk += wy[dj + 1] * r;
        }
        s += wz[dk + 1] * pk;
    }
    return s;
}

template <>
__device__ inline void d_sample_velocity<float, float>(
    const float* uu, const float* vv, const float* ww, float x, float y, float z, int nx, int ny,
    int nz, float dx, float dy, float dz, float Lx, float Ly, float Lz, float& vu, float& vvel,
    float& vw) {
    x  = tclamp<float>(x, 0.f, Lx);
    y  = tclamp<float>(y, 0.f, Ly);
    z  = tclamp<float>(z, 0.f, Lz);
    vu = d_face_value_f32(uu, 0, x / dx, y / dy + 0.5f, z / dz + 0.5f, nx, ny, nz, 0, 1, 1);
    vvel = d_face_value_f32(vv, 1, x / dx + 0.5f, y / dy, z / dz + 0.5f, nx, ny, nz, 1, 0, 1);
    vw   = d_face_value_f32(ww, 2, x / dx + 0.5f, y / dy + 0.5f, z / dz, nx, ny, nz, 1, 1, 0);
}

// One MAC face's separable value+gradient interpolation. Outputs Σ wx·wy·wz·val
// and the three analytic spline derivatives (still /dx-/dy-/dz scaled by caller).
// Partials: over di accumulate (A=Σwx·v, B=Σdwx·v); over dj accumulate
// (P=Σwy·A, Q=Σdwy·A, Rb=Σwy·B); over dk: v=Σwz·P, dx=Σwz·Rb, dy=Σwz·Q, dz=Σdwz·P.
__device__ inline void d_face_grad_f32(const float* __restrict fld, int axis, float cx, float cy,
                                       float cz, int nx, int ny, int nz, int ilo, int jlo, int klo,
                                       float& vout, float& gx, float& gy, float& gz) {
    int ic = t_round_idx<float>(cx), jc = t_round_idx<float>(cy), kc = t_round_idx<float>(cz);
    float wx[3], wy[3], wz[3], dwx[3], dwy[3], dwz[3];
    t_bspline<float>(cx - ic, wx);   t_bspline_d<float>(cx - ic, dwx);
    t_bspline<float>(cy - jc, wy);   t_bspline_d<float>(cy - jc, dwy);
    t_bspline<float>(cz - kc, wz);   t_bspline_d<float>(cz - kc, dwz);
    int ii0 = iclamp(ic - 1, ilo, nx), ii1 = iclamp(ic, ilo, nx), ii2 = iclamp(ic + 1, ilo, nx);
    int sj = (axis == 0) ? (nx + 1) : (nx + 2);
    int sk = (axis == 0) ? (nx + 1) * (ny + 2)
                         : (axis == 1 ? (nx + 2) * (ny + 1) : (nx + 2) * (ny + 2));
    float v = 0.f, dx = 0.f, dy = 0.f, dz = 0.f;
    for (int dk = -1; dk <= 1; dk++) {
        int kb = iclamp(kc + dk, klo, nz) * sk;
        float P = 0.f, Q = 0.f, Rb = 0.f; // dj-accumulated partials
        for (int dj = -1; dj <= 1; dj++) {
            int base = kb + iclamp(jc + dj, jlo, ny) * sj;
            float v0 = __ldg(&fld[base + ii0]);
            float v1 = __ldg(&fld[base + ii1]);
            float v2 = __ldg(&fld[base + ii2]);
            float A = wx[0] * v0 + wx[1] * v1 + wx[2] * v2;   // Σ_di wx·v
            float B = dwx[0] * v0 + dwx[1] * v1 + dwx[2] * v2; // Σ_di dwx·v
            float wyj = wy[dj + 1], dwyj = dwy[dj + 1];
            P  += wyj * A;
            Q  += dwyj * A;
            Rb += wyj * B;
        }
        float wzk = wz[dk + 1], dwzk = dwz[dk + 1];
        v  += wzk * P;
        dx += wzk * Rb;
        dy += wzk * Q;
        dz += dwzk * P;
    }
    vout = v; gx = dx; gy = dy; gz = dz;
}

template <>
__device__ inline void d_sample_velocity_grad<float, float>(
    const float* uu, const float* vv, const float* ww, float x, float y, float z, int nx, int ny,
    int nz, float dx, float dy, float dz, float Lx, float Ly, float Lz, float& vu, float& vvel,
    float& vw, float g[9]) {
    x = tclamp<float>(x, 0.f, Lx);
    y = tclamp<float>(y, 0.f, Ly);
    z = tclamp<float>(z, 0.f, Lz);
    float gx, gy, gz;
    d_face_grad_f32(uu, 0, x / dx, y / dy + 0.5f, z / dz + 0.5f, nx, ny, nz, 0, 1, 1, vu, gx, gy, gz);
    g[0] = gx / dx; g[1] = gy / dy; g[2] = gz / dz;
    d_face_grad_f32(vv, 1, x / dx + 0.5f, y / dy, z / dz + 0.5f, nx, ny, nz, 1, 0, 1, vvel, gx, gy, gz);
    g[3] = gx / dx; g[4] = gy / dy; g[5] = gz / dz;
    d_face_grad_f32(ww, 2, x / dx + 0.5f, y / dy + 0.5f, z / dz, nx, ny, nz, 1, 1, 0, vw, gx, gy, gz);
    g[6] = gx / dx; g[7] = gy / dy; g[8] = gz / dz;
}

// ════════════════════════════════════════════════════════════════════════════
// EXPERIMENT — FP16 (packed half2) samplers. Velocity scratch is pre-staged to
// __half; coordinates / index math stay FP32 (geometry needs the range). The
// inner 27-point B-spline runs in *genuine* packed half2:
//   * grad sampler packs the four per-tap weighted sums (value, ∂x, ∂y, ∂z) into
//     two __half2 lanes → 2 __hfma2 per tap instead of 4 scalar FMAs (≈2× ALU on
//     the packed FP16 pipe). This is the face_march lever (FMA-bound).
//   * plain sampler keeps a single sum per face but reads FP16 → halves the
//     gather bytes (the advect / face_pullback lever, memory-gather-bound).
// B-spline weights are computed in FP32 then narrowed to half2 (cheap, off the
// hot path). |u|~5, |∇u·dx|~O(1) — comfortably inside FP16 range.
// ════════════════════════════════════════════════════════════════════════════

// Plain velocity sample from FP16 scratch. Coords float; sum accumulated in float
// (single sum/face → no packing benefit, the win is the halved gather).
__device__ inline void d_sample_velocity_h16(const __half* uu, const __half* vv, const __half* ww,
                                             float x, float y, float z, int nx, int ny, int nz,
                                             float dx, float dy, float dz, float Lx, float Ly,
                                             float Lz, float& vu, float& vvel, float& vw) {
    x = tclamp<float>(x, 0.f, Lx);
    y = tclamp<float>(y, 0.f, Ly);
    z = tclamp<float>(z, 0.f, Lz);
    {
        float cx = x / dx, cy = y / dy + 0.5f, cz = z / dz + 0.5f;
        int ic = t_round_idx<float>(cx), jc = t_round_idx<float>(cy), kc = t_round_idx<float>(cz);
        float wx[3], wy[3], wz[3];
        t_bspline<float>(cx - ic, wx); t_bspline<float>(cy - jc, wy); t_bspline<float>(cz - kc, wz);
        float s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 0, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] *
                         __half2float(__ldg(&uu[lfm_iu(ii, jj, kk, nx, ny)]));
                }
        vu = s;
    }
    {
        float cx = x / dx + 0.5f, cy = y / dy, cz = z / dz + 0.5f;
        int ic = t_round_idx<float>(cx), jc = t_round_idx<float>(cy), kc = t_round_idx<float>(cz);
        float wx[3], wy[3], wz[3];
        t_bspline<float>(cx - ic, wx); t_bspline<float>(cy - jc, wy); t_bspline<float>(cz - kc, wz);
        float s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 0, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] *
                         __half2float(__ldg(&vv[lfm_iv(ii, jj, kk, nx, ny)]));
                }
        vvel = s;
    }
    {
        float cx = x / dx + 0.5f, cy = y / dy + 0.5f, cz = z / dz;
        int ic = t_round_idx<float>(cx), jc = t_round_idx<float>(cy), kc = t_round_idx<float>(cz);
        float wx[3], wy[3], wz[3];
        t_bspline<float>(cx - ic, wx); t_bspline<float>(cy - jc, wy); t_bspline<float>(cz - kc, wz);
        float s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 0, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] *
                         __half2float(__ldg(&ww[lfm_iw(ii, jj, kk, nx, ny)]));
                }
        vw = s;
    }
}

// One MAC-face contribution of the grad sampler, separable form. Values are read
// from FP16 scratch (the gather is the FP16 memory win) but the B-spline algebra
// runs separably in FP32 — full-rate FMA on the 3090 and far fewer ops than the
// packed-half2 27-tap version, which (post the separable FP32 rewrite) was the
// slower path. Partials: di→(A,B); dj→(P,Q,Rb); dk→(v,∂x,∂y,∂z).
__device__ inline void d_face_grad_h16(const __half* fld, int icoff, int jcoff, int kcoff,
                                       float cx, float cy, float cz, int nx, int ny, int nz,
                                       int ilo, int jlo, int klo, int axis,
                                       float& vout, float& gx, float& gy, float& gz) {
    int ic = t_round_idx<float>(cx), jc = t_round_idx<float>(cy), kc = t_round_idx<float>(cz);
    float wx[3], wy[3], wz[3], dwx[3], dwy[3], dwz[3];
    t_bspline<float>(cx - ic, wx);   t_bspline_d<float>(cx - ic, dwx);
    t_bspline<float>(cy - jc, wy);   t_bspline_d<float>(cy - jc, dwy);
    t_bspline<float>(cz - kc, wz);   t_bspline_d<float>(cz - kc, dwz);
    int ii0 = iclamp(ic - 1, ilo, nx), ii1 = iclamp(ic, ilo, nx), ii2 = iclamp(ic + 1, ilo, nx);
    int sj = (axis == 0) ? (nx + 1) : (nx + 2);
    int sk = (axis == 0) ? (nx + 1) * (ny + 2)
                         : (axis == 1 ? (nx + 2) * (ny + 1) : (nx + 2) * (ny + 2));
    float v = 0.f, dx = 0.f, dy = 0.f, dz = 0.f;
    for (int dk = -1; dk <= 1; dk++) {
        int kb = iclamp(kc + dk, klo, nz) * sk;
        float P = 0.f, Q = 0.f, Rb = 0.f;
        for (int dj = -1; dj <= 1; dj++) {
            int base = kb + iclamp(jc + dj, jlo, ny) * sj;
            float v0 = __half2float(__ldg(&fld[base + ii0]));
            float v1 = __half2float(__ldg(&fld[base + ii1]));
            float v2 = __half2float(__ldg(&fld[base + ii2]));
            float A = wx[0] * v0 + wx[1] * v1 + wx[2] * v2;
            float B = dwx[0] * v0 + dwx[1] * v1 + dwx[2] * v2;
            float wyj = wy[dj + 1], dwyj = dwy[dj + 1];
            P  += wyj * A;
            Q  += dwyj * A;
            Rb += wyj * B;
        }
        float wzk = wz[dk + 1], dwzk = dwz[dk + 1];
        v  += wzk * P;
        dx += wzk * Rb;
        dy += wzk * Q;
        dz += dwzk * P;
    }
    vout = v; gx = dx; gy = dy; gz = dz;
    (void)icoff; (void)jcoff; (void)kcoff;
}

// Velocity + 3×3 gradient from FP16 scratch, packed half2 inner loop. Mirrors
// d_sample_velocity_grad's MAC offsets/clamps; g[3a+b]=∂u_a/∂x_b.
__device__ inline void d_sample_velocity_grad_h16(const __half* uu, const __half* vv,
                                                  const __half* ww, float x, float y, float z,
                                                  int nx, int ny, int nz, float dx, float dy,
                                                  float dz, float Lx, float Ly, float Lz, float& vu,
                                                  float& vvel, float& vw, float g[9]) {
    x = tclamp<float>(x, 0.f, Lx);
    y = tclamp<float>(y, 0.f, Ly);
    z = tclamp<float>(z, 0.f, Lz);
    float gx, gy, gz;
    // u-face (clamps 0,1,1)
    d_face_grad_h16(uu, 0, 0, 0, x / dx, y / dy + 0.5f, z / dz + 0.5f, nx, ny, nz, 0, 1, 1, 0, vu,
                    gx, gy, gz);
    g[0] = gx / dx; g[1] = gy / dy; g[2] = gz / dz;
    // v-face (clamps 1,0,1)
    d_face_grad_h16(vv, 0, 0, 0, x / dx + 0.5f, y / dy, z / dz + 0.5f, nx, ny, nz, 1, 0, 1, 1, vvel,
                    gx, gy, gz);
    g[3] = gx / dx; g[4] = gy / dy; g[5] = gz / dz;
    // w-face (clamps 1,1,0)
    d_face_grad_h16(ww, 0, 0, 0, x / dx + 0.5f, y / dy + 0.5f, z / dz, nx, ny, nz, 1, 1, 0, 2, vw,
                    gx, gy, gz);
    g[6] = gx / dx; g[7] = gy / dy; g[8] = gz / dz;
}

// Specializations so the existing templated kernels (advect / face_march /
// face_pullback) transparently dispatch to the FP16 path when instantiated with
// VT=__half (T=float). Same signature as the primary templates.
template <>
__device__ inline void d_sample_velocity<float, __half>(
    const __half* uu, const __half* vv, const __half* ww, float x, float y, float z, int nx, int ny,
    int nz, float dx, float dy, float dz, float Lx, float Ly, float Lz, float& vu, float& vvel,
    float& vw) {
    d_sample_velocity_h16(uu, vv, ww, x, y, z, nx, ny, nz, dx, dy, dz, Lx, Ly, Lz, vu, vvel, vw);
}
template <>
__device__ inline void d_sample_velocity_grad<float, __half>(
    const __half* uu, const __half* vv, const __half* ww, float x, float y, float z, int nx, int ny,
    int nz, float dx, float dy, float dz, float Lx, float Ly, float Lz, float& vu, float& vvel,
    float& vw, float g[9]) {
    d_sample_velocity_grad_h16(uu, vv, ww, x, y, z, nx, ny, nz, dx, dy, dz, Lx, Ly, Lz, vu, vvel, vw,
                               g);
}

// 27-point quadratic B-spline at cell centers (interior-indexed scalar fields).
__device__ inline void d_sample_cell_centered(const double* sx, const double* sy, const double* sz,
                                              double x, double y, double z, int nx, int ny, int nz,
                                              double dx, double dy, double dz, double Lx, double Ly,
                                              double Lz, double& vx, double& vy, double& vz) {
    x = dclamp(x, 0.0, Lx);
    y = dclamp(y, 0.0, Ly);
    z = dclamp(z, 0.0, Lz);
    double cix = x / dx + 0.5, ciy = y / dy + 0.5, ciz = z / dz + 0.5;
    int ic = (int)floor(cix + 0.5), jc = (int)floor(ciy + 0.5), kc = (int)floor(ciz + 0.5);
    double wx[3], wy[3], wz[3];
    d_bspline(cix - ic, wx);
    d_bspline(ciy - jc, wy);
    d_bspline(ciz - kc, wz);
    vx = vy = vz = 0;
    for (int dk = -1; dk <= 1; dk++) {
        int kk = iclamp(kc + dk, 1, nz);
        for (int dj = -1; dj <= 1; dj++) {
            int jj = iclamp(jc + dj, 1, ny);
            for (int di = -1; di <= 1; di++) {
                int ii   = iclamp(ic + di, 1, nx);
                double w = wx[di + 1] * wy[dj + 1] * wz[dk + 1];
                long m   = lfm_fm(ii, jj, kk, nx, ny);
                vx += w * sx[m];
                vy += w * sy[m];
                vz += w * sz[m];
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// State allocate / free / transfer
// ──────────────────────────────────────────────────────────────────
void CudaLFMState3D::allocate(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_,
                              int n_steps_) {
    nx = nx_;
    ny = ny_;
    nz = nz_;
    dx = dx_;
    dy = dy_;
    dz = dz_;
    Lx = nx * dx;
    Ly = ny * dy;
    Lz = nz * dz;
    n_steps = n_steps_;

    g_.allocate(nx, ny, nz, dx, dy, dz);
    int ps = lfm_p_size(nx, ny, nz);
    d_p    = dmalloc(ps);
    d_rhs  = dmalloc(ps);

    int us = lfm_u_size(nx, ny, nz), vs = lfm_v_size(nx, ny, nz), ws = lfm_w_size(nx, ny, nz);
    auto alloc_vel = [&](CudaVel3D& q) {
        q.u = dmalloc(us);
        q.v = dmalloc(vs);
        q.w = dmalloc(ws);
    };
    alloc_vel(cur);
    alloc_vel(u0);
    alloc_vel(A);
    alloc_vel(B);
    alloc_vel(C);
    vb.resize(n_steps);
    for (auto& q : vb)
        alloc_vel(q);

    long fs = lfm_fm_size(nx, ny, nz);
    phi_x   = dmalloc(fs);
    phi_y   = dmalloc(fs);
    phi_z   = dmalloc(fs);
    psi_x   = dmalloc(fs);
    psi_y   = dmalloc(fs);
    psi_z   = dmalloc(fs);
    phi_mid_x = dmalloc(fs);
    phi_mid_y = dmalloc(fs);
    phi_mid_z = dmalloc(fs);
    for (int a = 0; a < 9; a++) {
        F[a]     = dmalloc(fs);
        T[a]     = dmalloc(fs);
        F_mid[a] = dmalloc(fs);
    }
    m_x    = dmalloc(fs);
    m_y    = dmalloc(fs);
    m_z    = dmalloc(fs);
    visc_x = dmalloc(fs);
    visc_y = dmalloc(fs);
    visc_z = dmalloc(fs);
    e_x    = dmalloc(fs);
    e_y    = dmalloc(fs);
    e_z    = dmalloc(fs);
    uhat_x = dmalloc(fs);
    uhat_y = dmalloc(fs);
    uhat_z = dmalloc(fs);

    // FIX①: per-axis face flow maps + face-sized impulse / scratch (MAC sizes).
    auto alloc_face_map = [&](CudaFaceFlowMap& f, int n) {
        for (double** p : {&f.bx, &f.by, &f.bz, &f.t0, &f.t1, &f.t2, &f.fx, &f.fy, &f.fz, &f.f0,
                           &f.f1, &f.f2})
            *p = dmalloc(n);
    };
    alloc_face_map(fmu, us);
    alloc_face_map(fmv, vs);
    alloc_face_map(fmw, ws);
    alloc_vel(mface);
    alloc_vel(mhat);
    alloc_vel(merr);

    // FP32 velocity/source scratch (one MAC field each) for the fp32 sampling path.
    vu32 = fmalloc(us);
    vv32 = fmalloc(vs);
    vw32 = fmalloc(ws);
    su32 = fmalloc(us);
    sv32 = fmalloc(vs);
    sw32 = fmalloc(ws);

    // FP16 sampling scratch (experiment). Always allocated (cheap: half the FP32
    // scratch bytes); only used when sample_fp16 is enabled at runtime.
    vu16 = hmalloc(us);
    vv16 = hmalloc(vs);
    vw16 = hmalloc(ws);
    su16 = hmalloc(us);
    sv16 = hmalloc(vs);
    sw16 = hmalloc(ws);
}

void CudaLFMState3D::free() {
    g_.free();
    dfree(d_p);
    dfree(d_rhs);
    auto free_vel = [&](CudaVel3D& q) {
        dfree(q.u);
        dfree(q.v);
        dfree(q.w);
    };
    free_vel(cur);
    free_vel(u0);
    free_vel(A);
    free_vel(B);
    free_vel(C);
    for (auto& q : vb)
        free_vel(q);
    vb.clear();
    dfree(phi_x);
    dfree(phi_y);
    dfree(phi_z);
    dfree(psi_x);
    dfree(psi_y);
    dfree(psi_z);
    dfree(phi_mid_x);
    dfree(phi_mid_y);
    dfree(phi_mid_z);
    for (int a = 0; a < 9; a++) {
        dfree(F[a]);
        dfree(T[a]);
        dfree(F_mid[a]);
    }
    dfree(m_x);
    dfree(m_y);
    dfree(m_z);
    dfree(visc_x);
    dfree(visc_y);
    dfree(visc_z);
    dfree(e_x);
    dfree(e_y);
    dfree(e_z);
    dfree(uhat_x);
    dfree(uhat_y);
    dfree(uhat_z);
    auto free_face_map = [&](CudaFaceFlowMap& f) {
        for (double** p : {&f.bx, &f.by, &f.bz, &f.t0, &f.t1, &f.t2, &f.fx, &f.fy, &f.fz, &f.f0,
                           &f.f1, &f.f2})
            dfree(*p);
    };
    free_face_map(fmu);
    free_face_map(fmv);
    free_face_map(fmw);
    free_vel(mface);
    free_vel(mhat);
    free_vel(merr);
    ffree(vu32);
    ffree(vv32);
    ffree(vw32);
    ffree(su32);
    ffree(sv32);
    ffree(sw32);
    hfree(vu16);
    hfree(vv16);
    hfree(vw16);
    hfree(su16);
    hfree(sv16);
    hfree(sw16);
}

void CudaLFMState3D::upload_velocity(const std::vector<double>& hu, const std::vector<double>& hv,
                                     const std::vector<double>& hw) {
    cudaMemcpy(cur.u, hu.data(), hu.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cur.v, hv.data(), hv.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cur.w, hw.data(), hw.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void CudaLFMState3D::download_velocity(std::vector<double>& hu, std::vector<double>& hv,
                                       std::vector<double>& hw) const {
    cudaMemcpy(hu.data(), cur.u, hu.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hv.data(), cur.v, hv.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hw.data(), cur.w, hw.size() * sizeof(double), cudaMemcpyDeviceToHost);
}

void CudaLFMState3D::upload_solid(const std::vector<char>& hsolid) {
    cudaMemcpy(g_.solid, hsolid.data(), hsolid.size() * sizeof(bool), cudaMemcpyHostToDevice);
}

void CudaLFMState3D::upload_to(CudaVel3D q, const std::vector<double>& hu,
                               const std::vector<double>& hv,
                               const std::vector<double>& hw) const {
    cudaMemcpy(q.u, hu.data(), hu.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(q.v, hv.data(), hv.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(q.w, hw.data(), hw.size() * sizeof(double), cudaMemcpyHostToDevice);
}

void CudaLFMState3D::download_from(CudaVel3D q, std::vector<double>& hu, std::vector<double>& hv,
                                   std::vector<double>& hw) const {
    cudaMemcpy(hu.data(), q.u, hu.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hv.data(), q.v, hv.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hw.data(), q.w, hw.size() * sizeof(double), cudaMemcpyDeviceToHost);
}

// ──────────────────────────────────────────────────────────────────
// Free-slip box BC — three ordered kernels (x → y → z) matching the CPU
// FreeSlipAllFaces3D + NoSlipImmersedSolid3D write order in patches_3d.cpp.
// ──────────────────────────────────────────────────────────────────
__global__ void bc_x_kernel(double* u, double* v, double* w, int nx, int ny, int nz) {
    int jj = blockIdx.x * blockDim.x + threadIdx.x; // 0..ny+1
    int kk = blockIdx.y * blockDim.y + threadIdx.y; // 0..nz+1
    if (jj > ny + 1 || kk > nz + 1)
        return;
    if (jj >= 1 && jj <= ny && kk >= 1 && kk <= nz) {
        u[lfm_iu(0, jj, kk, nx, ny)]  = 0.0;
        u[lfm_iu(nx, jj, kk, nx, ny)] = 0.0;
    }
    if (kk >= 1 && kk <= nz && jj >= 0 && jj <= ny) {
        v[lfm_iv(0, jj, kk, nx, ny)]      = v[lfm_iv(1, jj, kk, nx, ny)];
        v[lfm_iv(nx + 1, jj, kk, nx, ny)] = v[lfm_iv(nx, jj, kk, nx, ny)];
    }
    if (kk >= 1 && kk <= nz && jj >= 1 && jj <= ny) {
        w[lfm_iw(0, jj, kk, nx, ny)]      = w[lfm_iw(1, jj, kk, nx, ny)];
        w[lfm_iw(nx + 1, jj, kk, nx, ny)] = w[lfm_iw(nx, jj, kk, nx, ny)];
    }
}
__global__ void bc_y_kernel(double* u, double* v, double* w, int nx, int ny, int nz) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x; // 0..nx+1
    int kk = blockIdx.y * blockDim.y + threadIdx.y; // 1..nz
    if (ii > nx + 1 || kk > nz)
        return;
    if (kk < 1)
        return;
    if (ii >= 1 && ii <= nx) {
        v[lfm_iv(ii, 0, kk, nx, ny)]  = 0.0;
        v[lfm_iv(ii, ny, kk, nx, ny)] = 0.0;
        w[lfm_iw(ii, 0, kk, nx, ny)]      = w[lfm_iw(ii, 1, kk, nx, ny)];
        w[lfm_iw(ii, ny + 1, kk, nx, ny)] = w[lfm_iw(ii, ny, kk, nx, ny)];
    }
    if (ii >= 0 && ii <= nx) {
        u[lfm_iu(ii, 0, kk, nx, ny)]      = u[lfm_iu(ii, 1, kk, nx, ny)];
        u[lfm_iu(ii, ny + 1, kk, nx, ny)] = u[lfm_iu(ii, ny, kk, nx, ny)];
    }
}
__global__ void bc_z_kernel(double* u, double* v, double* w, int nx, int ny, int nz) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x; // 0..nx+1
    int jj = blockIdx.y * blockDim.y + threadIdx.y; // 1..ny
    if (ii > nx + 1 || jj > ny)
        return;
    if (jj < 1)
        return;
    if (ii >= 1 && ii <= nx) {
        w[lfm_iw(ii, jj, 0, nx, ny)]  = 0.0;
        w[lfm_iw(ii, jj, nz, nx, ny)] = 0.0;
        v[lfm_iv(ii, jj, 0, nx, ny)]      = v[lfm_iv(ii, jj, 1, nx, ny)];
        v[lfm_iv(ii, jj, nz + 1, nx, ny)] = v[lfm_iv(ii, jj, nz, nx, ny)];
    }
    if (ii >= 0 && ii <= nx) {
        u[lfm_iu(ii, jj, 0, nx, ny)]      = u[lfm_iu(ii, jj, 1, nx, ny)];
        u[lfm_iu(ii, jj, nz + 1, nx, ny)] = u[lfm_iu(ii, jj, nz, nx, ny)];
    }
}

// No-slip immersed solid: zero faces between a solid cell and a fluid neighbour.
__global__ void bc_solid_kernel(double* u, double* v, double* w, const bool* solid, int nx, int ny,
                                 int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    if (!solid[lfm_ip(i, j, k, nx, ny)])
        return;
    if (i > 1 && !solid[lfm_ip(i - 1, j, k, nx, ny)])
        u[lfm_iu(i - 1, j, k, nx, ny)] = 0.0;
    if (i < nx && !solid[lfm_ip(i + 1, j, k, nx, ny)])
        u[lfm_iu(i, j, k, nx, ny)] = 0.0;
    if (j > 1 && !solid[lfm_ip(i, j - 1, k, nx, ny)])
        v[lfm_iv(i, j - 1, k, nx, ny)] = 0.0;
    if (j < ny && !solid[lfm_ip(i, j + 1, k, nx, ny)])
        v[lfm_iv(i, j, k, nx, ny)] = 0.0;
    if (k > 1 && !solid[lfm_ip(i, j, k - 1, nx, ny)])
        w[lfm_iw(i, j, k - 1, nx, ny)] = 0.0;
    if (k < nz && !solid[lfm_ip(i, j, k + 1, nx, ny)])
        w[lfm_iw(i, j, k, nx, ny)] = 0.0;
}

void lfm_apply_free_slip_box(CudaLFMState3D& s, CudaVel3D vel) {
    int nx = s.nx, ny = s.ny, nz = s.nz;
    dim3 b(16, 16);
    // Grids must cover the full ghost-inclusive index ranges each kernel touches
    // (up to nx+1/ny+1/nz+1); the per-thread guards drop the excess. Undersizing
    // here silently skips the far wall row (j=ny / k=nz), leaving a non-zero
    // wall-normal velocity → spurious divergence.
    auto gd = [](int n) { return (n + 2 + 15) / 16; };
    bc_x_kernel<<<dim3(gd(ny), gd(nz)), b>>>(vel.u, vel.v, vel.w, nx, ny, nz);
    bc_y_kernel<<<dim3(gd(nx), gd(nz)), b>>>(vel.u, vel.v, vel.w, nx, ny, nz);
    bc_z_kernel<<<dim3(gd(nx), gd(ny)), b>>>(vel.u, vel.v, vel.w, nx, ny, nz);
    bc_solid_kernel<<<grid3(nx, ny, nz), block3()>>>(vel.u, vel.v, vel.w, s.g_.solid, nx, ny, nz);
}

// ──────────────────────────────────────────────────────────────────
// Freestream box BC — prescribe the SAME freestream (Ux,Uy,Uz) on all six
// walls (normal velocity + tangential ghost) so inflow flux = outflow flux
// (mass-balanced, pure-Neumann). Mirrors the CPU FreestreamBox3D. + no-slip
// immersed solid. Used for the delta wing (paper Fig. 9).
// ──────────────────────────────────────────────────────────────────
__global__ void bc_x_freestream_kernel(double* u, double* v, double* w, int nx, int ny, int nz,
                                        double Ux, double Uy, double Uz) {
    int jj = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    if (jj > ny + 1 || kk > nz + 1)
        return;
    if (jj >= 1 && jj <= ny && kk >= 1 && kk <= nz) {
        u[lfm_iu(0, jj, kk, nx, ny)]  = Ux;
        u[lfm_iu(nx, jj, kk, nx, ny)] = Ux;
    }
    if (kk >= 1 && kk <= nz && jj >= 0 && jj <= ny) {
        v[lfm_iv(0, jj, kk, nx, ny)]      = Uy;
        v[lfm_iv(nx + 1, jj, kk, nx, ny)] = Uy;
    }
    if (kk >= 1 && kk <= nz && jj >= 1 && jj <= ny) {
        w[lfm_iw(0, jj, kk, nx, ny)]      = Uz;
        w[lfm_iw(nx + 1, jj, kk, nx, ny)] = Uz;
    }
}
__global__ void bc_y_freestream_kernel(double* u, double* v, double* w, int nx, int ny, int nz,
                                        double Ux, double Uy, double Uz) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii > nx + 1 || kk > nz || kk < 1)
        return;
    if (ii >= 1 && ii <= nx) {
        v[lfm_iv(ii, 0, kk, nx, ny)]      = Uy;
        v[lfm_iv(ii, ny, kk, nx, ny)]     = Uy;
        w[lfm_iw(ii, 0, kk, nx, ny)]      = Uz;
        w[lfm_iw(ii, ny + 1, kk, nx, ny)] = Uz;
    }
    if (ii >= 0 && ii <= nx) {
        u[lfm_iu(ii, 0, kk, nx, ny)]      = Ux;
        u[lfm_iu(ii, ny + 1, kk, nx, ny)] = Ux;
    }
}
__global__ void bc_z_freestream_kernel(double* u, double* v, double* w, int nx, int ny, int nz,
                                        double Ux, double Uy, double Uz) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii > nx + 1 || jj > ny || jj < 1)
        return;
    if (ii >= 1 && ii <= nx) {
        w[lfm_iw(ii, jj, 0, nx, ny)]      = Uz;
        w[lfm_iw(ii, jj, nz, nx, ny)]     = Uz;
        v[lfm_iv(ii, jj, 0, nx, ny)]      = Uy;
        v[lfm_iv(ii, jj, nz + 1, nx, ny)] = Uy;
    }
    if (ii >= 0 && ii <= nx) {
        u[lfm_iu(ii, jj, 0, nx, ny)]      = Ux;
        u[lfm_iu(ii, jj, nz + 1, nx, ny)] = Ux;
    }
}
void lfm_apply_freestream_box(CudaLFMState3D& s, CudaVel3D vel, double Ux, double Uy, double Uz) {
    int nx = s.nx, ny = s.ny, nz = s.nz;
    dim3 b(16, 16);
    auto gd = [](int n) { return (n + 2 + 15) / 16; };
    bc_x_freestream_kernel<<<dim3(gd(ny), gd(nz)), b>>>(vel.u, vel.v, vel.w, nx, ny, nz, Ux, Uy, Uz);
    bc_y_freestream_kernel<<<dim3(gd(nx), gd(nz)), b>>>(vel.u, vel.v, vel.w, nx, ny, nz, Ux, Uy, Uz);
    bc_z_freestream_kernel<<<dim3(gd(nx), gd(ny)), b>>>(vel.u, vel.v, vel.w, nx, ny, nz, Ux, Uy, Uz);
    bc_solid_kernel<<<grid3(nx, ny, nz), block3()>>>(vel.u, vel.v, vel.w, s.g_.solid, nx, ny, nz);
}

// ──────────────────────────────────────────────────────────────────
// Pressure projection: rhs = ∇·u/dt → CudaPCG3D::solve → u -= dt·∇p.
// Mirrors PressureProjection3D::project (src/numerics/pressure/pressure_3d.cpp).
// ──────────────────────────────────────────────────────────────────
__global__ void build_divergence_rhs_kernel(const double* u, const double* v, const double* w,
                                             const bool* solid, double* rhs, int nx, int ny, int nz,
                                             double dx, double dy, double dz, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    int cid = lfm_ip(i, j, k, nx, ny);
    if (solid[cid]) {
        rhs[cid] = 0.0;
        return;
    }
    double div = (u[lfm_iu(i, j, k, nx, ny)] - u[lfm_iu(i - 1, j, k, nx, ny)]) / dx +
                 (v[lfm_iv(i, j, k, nx, ny)] - v[lfm_iv(i, j - 1, k, nx, ny)]) / dy +
                 (w[lfm_iw(i, j, k, nx, ny)] - w[lfm_iw(i, j, k - 1, nx, ny)]) / dz;
    rhs[cid] = div / dt;
}

__global__ void correct_velocity_kernel(double* u, double* v, double* w, const double* p,
                                         const bool* solid, int nx, int ny, int nz, double dx,
                                         double dy, double dz, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > nx || j > ny || k > nz)
        return;
    if (i < nx && !solid[lfm_ip(i, j, k, nx, ny)] && !solid[lfm_ip(i + 1, j, k, nx, ny)])
        u[lfm_iu(i, j, k, nx, ny)] -=
            dt * (p[lfm_ip(i + 1, j, k, nx, ny)] - p[lfm_ip(i, j, k, nx, ny)]) / dx;
    if (j < ny && !solid[lfm_ip(i, j, k, nx, ny)] && !solid[lfm_ip(i, j + 1, k, nx, ny)])
        v[lfm_iv(i, j, k, nx, ny)] -=
            dt * (p[lfm_ip(i, j + 1, k, nx, ny)] - p[lfm_ip(i, j, k, nx, ny)]) / dy;
    if (k < nz && !solid[lfm_ip(i, j, k, nx, ny)] && !solid[lfm_ip(i, j, k + 1, nx, ny)])
        w[lfm_iw(i, j, k, nx, ny)] -=
            dt * (p[lfm_ip(i, j, k + 1, nx, ny)] - p[lfm_ip(i, j, k, nx, ny)]) / dz;
}

void lfm_project(CudaLFMState3D& s, CudaVel3D vel, double dt, int iters, double tol) {
    int nx = s.nx, ny = s.ny, nz = s.nz;
    int ps = lfm_p_size(nx, ny, nz);
    cudaMemset(s.d_rhs, 0, (size_t)ps * sizeof(double));
    build_divergence_rhs_kernel<<<grid3(nx, ny, nz), block3()>>>(
        vel.u, vel.v, vel.w, s.g_.solid, s.d_rhs, nx, ny, nz, s.dx, s.dy, s.dz, dt);
    cudaMemset(s.d_p, 0, (size_t)ps * sizeof(double));
    // Precision routing: default to the fully-FP32 tile-native solve (the author's
    // regime — under-converged 6-iter solves don't need FP64). Set PCG_FP64=1 to
    // fall back to the high-accuracy FP64 tile-native solve.
    static const bool pcg_fp64 = std::getenv("PCG_FP64") != nullptr;
    if (pcg_fp64)
        s.pcg_.solve(s.g_, s.d_p, s.d_rhs, iters, tol);
    else
        s.pcg_.solve_f32_tile(s.g_, s.d_p, s.d_rhs, iters, tol);
    correct_velocity_kernel<<<grid3(nx, ny, nz), block3()>>>(
        vel.u, vel.v, vel.w, s.d_p, s.g_.solid, nx, ny, nz, s.dx, s.dy, s.dz, dt);
}

// ──────────────────────────────────────────────────────────────────
// RK2 semi-Lagrangian advection (mirrors LFMSimulator3D::rk2_advect).
// Backtrace each face with vel (midpoint), then sample src at the foot.
// ──────────────────────────────────────────────────────────────────
struct GeomParams {
    int nx, ny, nz;
    double dx, dy, dz, Lx, Ly, Lz;
};

// Pointer bundle for one flow map (Φ,F) or (Ψ,T): position + 9 Jacobian comps.
struct FMPtrs {
    double *px, *py, *pz;
    double* J[9];
};

template <typename T, typename VT>
__device__ inline void d_backtrace(const VT* vu, const VT* vv, const VT* vw, T x,
                                    T y, T z, T dt_step, const GeomParams g,
                                    T& xo, T& yo, T& zo) {
    T gdx = (T)g.dx, gdy = (T)g.dy, gdz = (T)g.dz, gLx = (T)g.Lx, gLy = (T)g.Ly, gLz = (T)g.Lz;
    T u1, v1, w1;
    d_sample_velocity<T, VT>(vu, vv, vw, x, y, z, g.nx, g.ny, g.nz, gdx, gdy, gdz, gLx, gLy, gLz, u1,
                         v1, w1);
    T xm = x - T(0.5) * dt_step * u1, ym = y - T(0.5) * dt_step * v1, zm = z - T(0.5) * dt_step * w1;
    T um, vm, wm;
    d_sample_velocity<T, VT>(vu, vv, vw, xm, ym, zm, g.nx, g.ny, g.nz, gdx, gdy, gdz, gLx, gLy, gLz,
                         um, vm, wm);
    xo = x - dt_step * um;
    yo = y - dt_step * vm;
    zo = z - dt_step * wm;
}

template <typename T, typename VT>
__global__ void advect_u_kernel(double* dst, const VT* su, const VT* sv, const VT* sw,
                                 const VT* vu, const VT* vv, const VT* vw,
                                 const bool* solid, double dt_step, GeomParams g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i >= g.nx || j > g.ny || k > g.nz)
        return; // i in [1, nx-1]
    int id = lfm_iu(i, j, k, g.nx, g.ny);
    if (solid[lfm_ip(i, j, k, g.nx, g.ny)] || solid[lfm_ip(i + 1, j, k, g.nx, g.ny)]) {
        dst[id] = 0;
        return;
    }
    T xs, ys, zs, ou, ov, ow, dts = (T)dt_step;
    d_backtrace<T, VT>(vu, vv, vw, T(i * g.dx), T((j - 0.5) * g.dy), T((k - 0.5) * g.dz), dts, g, xs, ys, zs);
    d_sample_velocity<T, VT>(su, sv, sw, xs, ys, zs, g.nx, g.ny, g.nz, (T)g.dx, (T)g.dy, (T)g.dz, (T)g.Lx, (T)g.Ly, (T)g.Lz,
                         ou, ov, ow);
    dst[id] = ou;
}
template <typename T, typename VT>
__global__ void advect_v_kernel(double* dst, const VT* su, const VT* sv, const VT* sw,
                                 const VT* vu, const VT* vv, const VT* vw,
                                 const bool* solid, double dt_step, GeomParams g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > g.nx || j >= g.ny || k > g.nz)
        return; // j in [1, ny-1]
    int id = lfm_iv(i, j, k, g.nx, g.ny);
    if (solid[lfm_ip(i, j, k, g.nx, g.ny)] || solid[lfm_ip(i, j + 1, k, g.nx, g.ny)]) {
        dst[id] = 0;
        return;
    }
    T xs, ys, zs, ou, ov, ow, dts = (T)dt_step;
    d_backtrace<T, VT>(vu, vv, vw, T((i - 0.5) * g.dx), T(j * g.dy), T((k - 0.5) * g.dz), dts, g, xs, ys, zs);
    d_sample_velocity<T, VT>(su, sv, sw, xs, ys, zs, g.nx, g.ny, g.nz, (T)g.dx, (T)g.dy, (T)g.dz, (T)g.Lx, (T)g.Ly, (T)g.Lz,
                         ou, ov, ow);
    dst[id] = ov;
}
template <typename T, typename VT>
__global__ void advect_w_kernel(double* dst, const VT* su, const VT* sv, const VT* sw,
                                 const VT* vu, const VT* vv, const VT* vw,
                                 const bool* solid, double dt_step, GeomParams g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > g.nx || j > g.ny || k >= g.nz)
        return; // k in [1, nz-1]
    int id = lfm_iw(i, j, k, g.nx, g.ny);
    if (solid[lfm_ip(i, j, k, g.nx, g.ny)] || solid[lfm_ip(i, j, k + 1, g.nx, g.ny)]) {
        dst[id] = 0;
        return;
    }
    T xs, ys, zs, ou, ov, ow, dts = (T)dt_step;
    d_backtrace<T, VT>(vu, vv, vw, T((i - 0.5) * g.dx), T((j - 0.5) * g.dy), T(k * g.dz), dts, g, xs, ys, zs);
    d_sample_velocity<T, VT>(su, sv, sw, xs, ys, zs, g.nx, g.ny, g.nz, (T)g.dx, (T)g.dy, (T)g.dz, (T)g.Lx, (T)g.Ly, (T)g.Lz,
                         ou, ov, ow);
    dst[id] = ow;
}

static GeomParams geom(const CudaLFMState3D& s) {
    return GeomParams{s.nx, s.ny, s.nz, s.dx, s.dy, s.dz, s.Lx, s.Ly, s.Lz};
}

void lfm_rk2_advect(CudaLFMState3D& s, CudaVel3D dst, CudaVel3D src, CudaVel3D vel, double dt_step) {
    GeomParams g = geom(s);
    dim3 blk = block3(), gr = grid3(s.nx, s.ny, s.nz);
    if (fp16_sampling(s)) {
        int us = lfm_u_size(s.nx, s.ny, s.nz), vs = lfm_v_size(s.nx, s.ny, s.nz),
            ws = lfm_w_size(s.nx, s.ny, s.nz);
        __half *su16 = (__half*)s.su16, *sv16 = (__half*)s.sv16, *sw16 = (__half*)s.sw16;
        __half *vu16 = (__half*)s.vu16, *vv16 = (__half*)s.vv16, *vw16 = (__half*)s.vw16;
        convert_vel_f16(src, su16, sv16, sw16, us, vs, ws);
        convert_vel_f16(vel, vu16, vv16, vw16, us, vs, ws);
        advect_u_kernel<float, __half><<<gr, blk>>>(dst.u, su16, sv16, sw16, vu16, vv16, vw16, s.g_.solid, dt_step, g);
        advect_v_kernel<float, __half><<<gr, blk>>>(dst.v, su16, sv16, sw16, vu16, vv16, vw16, s.g_.solid, dt_step, g);
        advect_w_kernel<float, __half><<<gr, blk>>>(dst.w, su16, sv16, sw16, vu16, vv16, vw16, s.g_.solid, dt_step, g);
    } else if (s.fp32_march) {
        int us = lfm_u_size(s.nx, s.ny, s.nz), vs = lfm_v_size(s.nx, s.ny, s.nz),
            ws = lfm_w_size(s.nx, s.ny, s.nz);
        // Stage src + vel as FP32 once → the hot gather reads pure float (no per-load FP64 convert).
        convert_vel_f32(src, s.su32, s.sv32, s.sw32, us, vs, ws);
        convert_vel_f32(vel, s.vu32, s.vv32, s.vw32, us, vs, ws);
        advect_u_kernel<float, float><<<gr, blk>>>(dst.u, s.su32, s.sv32, s.sw32, s.vu32, s.vv32, s.vw32, s.g_.solid, dt_step, g);
        advect_v_kernel<float, float><<<gr, blk>>>(dst.v, s.su32, s.sv32, s.sw32, s.vu32, s.vv32, s.vw32, s.g_.solid, dt_step, g);
        advect_w_kernel<float, float><<<gr, blk>>>(dst.w, s.su32, s.sv32, s.sw32, s.vu32, s.vv32, s.vw32, s.g_.solid, dt_step, g);
    } else {
        advect_u_kernel<double, double><<<gr, blk>>>(dst.u, src.u, src.v, src.w, vel.u, vel.v, vel.w, s.g_.solid, dt_step, g);
        advect_v_kernel<double, double><<<gr, blk>>>(dst.v, src.u, src.v, src.w, vel.u, vel.v, vel.w, s.g_.solid, dt_step, g);
        advect_w_kernel<double, double><<<gr, blk>>>(dst.w, src.u, src.v, src.w, vel.u, vel.v, vel.w, s.g_.solid, dt_step, g);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Flow-map RK4 marching (mirrors LFMSimulator3D velocity_gradient / march_cell
// / rk4_march_forward / rk4_march_backward). g[9] = ∂u_a/∂x_b row-major.
// ══════════════════════════════════════════════════════════════════════
__device__ inline void d_velocity_gradient(int i, int j, int k, const double* ug, const double* vg,
                                           const double* wg, const bool* solid, GeomParams G,
                                           double g[9]) {
    int nx = G.nx, ny = G.ny, nz = G.nz;
    double dx = G.dx, dy = G.dy, dz = G.dz;
    int ip1 = min(i + 1, nx), im1 = max(i - 1, 1);
    int jp1 = min(j + 1, ny), jm1 = max(j - 1, 1);
    int kp1 = min(k + 1, nz), km1 = max(k - 1, 1);
    bool sxm = solid[lfm_ip(i - 1, j, k, nx, ny)], sxp = solid[lfm_ip(i + 1, j, k, nx, ny)];
    bool sym = solid[lfm_ip(i, j - 1, k, nx, ny)], syp = solid[lfm_ip(i, j + 1, k, nx, ny)];
    bool szm = solid[lfm_ip(i, j, k - 1, nx, ny)], szp = solid[lfm_ip(i, j, k + 1, nx, ny)];
    g[0] = (ug[lfm_iu(i, j, k, nx, ny)] - ug[lfm_iu(i - 1, j, k, nx, ny)]) / dx;
    g[4] = (vg[lfm_iv(i, j, k, nx, ny)] - vg[lfm_iv(i, j - 1, k, nx, ny)]) / dy;
    g[8] = (wg[lfm_iw(i, j, k, nx, ny)] - wg[lfm_iw(i, j, k - 1, nx, ny)]) / dz;
    g[1] = (sym || syp) ? 0.0
                        : (ug[lfm_iu(i, jp1, k, nx, ny)] - ug[lfm_iu(i, jm1, k, nx, ny)]) / (2 * dy);
    g[2] = (szm || szp) ? 0.0
                        : (ug[lfm_iu(i, j, kp1, nx, ny)] - ug[lfm_iu(i, j, km1, nx, ny)]) / (2 * dz);
    g[3] = (sxm || sxp) ? 0.0
                        : (vg[lfm_iv(ip1, j, k, nx, ny)] - vg[lfm_iv(im1, j, k, nx, ny)]) / (2 * dx);
    g[5] = (szm || szp) ? 0.0
                        : (vg[lfm_iv(i, j, kp1, nx, ny)] - vg[lfm_iv(i, j, km1, nx, ny)]) / (2 * dz);
    g[6] = (sxm || sxp) ? 0.0
                        : (wg[lfm_iw(ip1, j, k, nx, ny)] - wg[lfm_iw(im1, j, k, nx, ny)]) / (2 * dx);
    g[7] = (sym || syp) ? 0.0
                        : (wg[lfm_iw(i, jp1, k, nx, ny)] - wg[lfm_iw(i, jm1, k, nx, ny)]) / (2 * dy);
}

__device__ inline void d_velocity_gradient_at(double x, double y, double z, const double* ug,
                                               const double* vg, const double* wg,
                                               const bool* solid, GeomParams G, double g[9]) {
    int ci = iclamp((int)(x / G.dx + 0.5), 1, G.nx);
    int cj = iclamp((int)(y / G.dy + 0.5), 1, G.ny);
    int ck = iclamp((int)(z / G.dz + 0.5), 1, G.nz);
    d_velocity_gradient(ci, cj, ck, ug, vg, wg, solid, G, g);
}

// One RK4 step of dΦ/dt=u(Φ), dF/dt=∇u·F for a single cell (state s[12]).
__device__ inline void d_march_cell(double& px, double& py, double& pz, double F[9],
                                    const double* uu, const double* vv, const double* ww,
                                    const bool* solid, double dt_march, GeomParams G) {
    double s0[12] = {px, py, pz, F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8]};
    auto rhs = [&](const double s[12], double d[12]) {
        double vu, vvel, vw, g[9];
        // Velocity and ∇u from the SAME B-spline (analytic spline derivative), matching the
        // author's InterpMacN2Grad. The old path snapped to the nearest cell and finite-
        // differenced ∇u — inconsistent with the sampled u and over-dissipative.
        d_sample_velocity_grad(uu, vv, ww, s[0], s[1], s[2], G.nx, G.ny, G.nz, G.dx, G.dy, G.dz,
                               G.Lx, G.Ly, G.Lz, vu, vvel, vw, g);
        d[0] = vu;
        d[1] = vvel;
        d[2] = vw;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                d[3 + 3 * a + b] = g[3 * a + 0] * s[3 + 0 * 3 + b] +
                                   g[3 * a + 1] * s[3 + 1 * 3 + b] + g[3 * a + 2] * s[3 + 2 * 3 + b];
    };
    // Incremental RK4: keep only s0, the running increment acc = k1+2k2+2k3+k4,
    // the current stage's k, and the next stage's input tmp (≈48 doubles instead
    // of 84). The summation order is identical to the CPU reference, so with FMA
    // disabled the result stays bit-for-bit equal (validated by P2/P4).
    double acc[12], k[12], tmp[12];
    rhs(s0, k); // stage 1
    for (int m = 0; m < 12; m++) {
        k[m] *= dt_march;
        acc[m] = k[m];
        tmp[m] = s0[m] + 0.5 * k[m];
    }
    rhs(tmp, k); // stage 2
    for (int m = 0; m < 12; m++) {
        k[m] *= dt_march;
        acc[m] += 2.0 * k[m];
        tmp[m] = s0[m] + 0.5 * k[m];
    }
    rhs(tmp, k); // stage 3
    for (int m = 0; m < 12; m++) {
        k[m] *= dt_march;
        acc[m] += 2.0 * k[m];
        tmp[m] = s0[m] + k[m];
    }
    rhs(tmp, k); // stage 4
    for (int m = 0; m < 12; m++) {
        k[m] *= dt_march;
        acc[m] += k[m];
        acc[m] = s0[m] + acc[m] / 6.0; // reuse acc as the output state
    }
    px = dclamp(acc[0], 0.0, G.Lx);
    py = dclamp(acc[1], 0.0, G.Ly);
    pz = dclamp(acc[2], 0.0, G.Lz);
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
            double val = acc[3 + 3 * a + b];
            if (!isfinite(val))
                val = (a == b) ? 1.0 : 0.0;
            F[3 * a + b] = val;
        }
}

__global__ __launch_bounds__(256) void march_kernel(FMPtrs fm, const double* uu, const double* vv,
                                                    const double* ww, const bool* solid,
                                                    double dt_march, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    long m = lfm_fm(i, j, k, G.nx, G.ny);
    double F[9];
    for (int a = 0; a < 9; a++)
        F[a] = fm.J[a][m];
    double px = fm.px[m], py = fm.py[m], pz = fm.pz[m];
    d_march_cell(px, py, pz, F, uu, vv, ww, solid, dt_march, G);
    fm.px[m] = px;
    fm.py[m] = py;
    fm.pz[m] = pz;
    for (int a = 0; a < 9; a++)
        fm.J[a][m] = F[a];
}

__global__ void set_identity_kernel(FMPtrs fm, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    long m   = lfm_fm(i, j, k, G.nx, G.ny);
    fm.px[m] = (i - 0.5) * G.dx;
    fm.py[m] = (j - 0.5) * G.dy;
    fm.pz[m] = (k - 0.5) * G.dz;
    for (int a = 0; a < 9; a++)
        fm.J[a][m] = (a == 0 || a == 4 || a == 8) ? 1.0 : 0.0;
}

static FMPtrs fwd_ptrs(CudaLFMState3D& s) {
    FMPtrs f;
    f.px = s.phi_x;
    f.py = s.phi_y;
    f.pz = s.phi_z;
    for (int a = 0; a < 9; a++)
        f.J[a] = s.F[a];
    return f;
}
static FMPtrs bwd_ptrs(CudaLFMState3D& s) {
    FMPtrs f;
    f.px = s.psi_x;
    f.py = s.psi_y;
    f.pz = s.psi_z;
    for (int a = 0; a < 9; a++)
        f.J[a] = s.T[a];
    return f;
}

void lfm_set_identity(CudaLFMState3D& s) {
    set_identity_kernel<<<grid3(s.nx, s.ny, s.nz), block3()>>>(fwd_ptrs(s), geom(s));
}
void lfm_set_backward_identity(CudaLFMState3D& s) {
    set_identity_kernel<<<grid3(s.nx, s.ny, s.nz), block3()>>>(bwd_ptrs(s), geom(s));
}
// The RK4 march carries a 12-component state (3 position + 9 Jacobian) plus
// k1..k4/tmp/out, so it is register/local-memory heavy — a 512-thread block
// overflows the launch resource limit. Use a 4×4×4 (64-thread) block.
static dim3 block_march() { return dim3(8, 8, 4); } // 256 threads (matches __launch_bounds__)
static dim3 grid_march(int nx, int ny, int nz) {
    return dim3((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
}
void lfm_rk4_march_forward(CudaLFMState3D& s, CudaVel3D vel, double dt_march) {
    march_kernel<<<grid_march(s.nx, s.ny, s.nz), block_march()>>>(
        fwd_ptrs(s), vel.u, vel.v, vel.w, s.g_.solid, dt_march, geom(s));
}
void lfm_rk4_march_backward(CudaLFMState3D& s, CudaVel3D vel, double dt_march) {
    march_kernel<<<grid_march(s.nx, s.ny, s.nz), block_march()>>>(
        bwd_ptrs(s), vel.u, vel.v, vel.w, s.g_.solid, dt_march, geom(s));
}

// ══════════════════════════════════════════════════════════════════════
// P3: impulse chain — viscous force, accumulation, midpoints, pullback,
// forward pullback, error correction, gauge write-back. Each mirrors the
// matching routine in src/simulator/lfm_simulator_3d.cpp.
// ══════════════════════════════════════════════════════════════════════

// μ∇²u at cell centers, per component → visc_x/visc_y/visc_z (interior arrays).
__global__ void viscous_kernel(const double* u, const double* v, const double* w,
                               const bool* solid, double* vx, double* vy, double* vz, GeomParams G,
                               double mu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    int nx = G.nx, ny = G.ny, nz = G.nz;
    long m = lfm_fm(i, j, k, nx, ny);
    if (solid[lfm_ip(i, j, k, nx, ny)]) {
        vx[m] = vy[m] = vz[m] = 0.0;
        return;
    }
    double idx2 = 1.0 / (G.dx * G.dx), idy2 = 1.0 / (G.dy * G.dy), idz2 = 1.0 / (G.dz * G.dz);
#define U(a, b, c) u[lfm_iu(a, b, c, nx, ny)]
#define V(a, b, c) v[lfm_iv(a, b, c, nx, ny)]
#define W(a, b, c) w[lfm_iw(a, b, c, nx, ny)]
    double uc = 0.5 * (U(i, j, k) + U(i - 1, j, k));
    double uL = (i > 1) ? 0.5 * (U(i - 1, j, k) + U(i - 2, j, k)) : uc;
    double uR = (i < nx) ? 0.5 * (U(i + 1, j, k) + U(i, j, k)) : uc;
    double uB = (j > 1) ? 0.5 * (U(i, j - 1, k) + U(i - 1, j - 1, k)) : uc;
    double uT = (j < ny) ? 0.5 * (U(i, j + 1, k) + U(i - 1, j + 1, k)) : uc;
    double uF = (k > 1) ? 0.5 * (U(i, j, k - 1) + U(i - 1, j, k - 1)) : uc;
    double uK = (k < nz) ? 0.5 * (U(i, j, k + 1) + U(i - 1, j, k + 1)) : uc;
    vx[m] = mu * ((uL + uR - 2 * uc) * idx2 + (uB + uT - 2 * uc) * idy2 + (uF + uK - 2 * uc) * idz2);
    double vc = 0.5 * (V(i, j, k) + V(i, j - 1, k));
    double vL = (i > 1) ? 0.5 * (V(i - 1, j, k) + V(i - 1, j - 1, k)) : vc;
    double vR = (i < nx) ? 0.5 * (V(i + 1, j, k) + V(i + 1, j - 1, k)) : vc;
    double vB = (j > 1) ? 0.5 * (V(i, j - 1, k) + V(i, j - 2, k)) : vc;
    double vT = (j < ny) ? 0.5 * (V(i, j + 1, k) + V(i, j, k)) : vc;
    double vF = (k > 1) ? 0.5 * (V(i, j, k - 1) + V(i, j - 1, k - 1)) : vc;
    double vK = (k < nz) ? 0.5 * (V(i, j, k + 1) + V(i, j - 1, k + 1)) : vc;
    vy[m] = mu * ((vL + vR - 2 * vc) * idx2 + (vB + vT - 2 * vc) * idy2 + (vF + vK - 2 * vc) * idz2);
    double wc = 0.5 * (W(i, j, k) + W(i, j, k - 1));
    double wL = (i > 1) ? 0.5 * (W(i - 1, j, k) + W(i - 1, j, k - 1)) : wc;
    double wR = (i < nx) ? 0.5 * (W(i + 1, j, k) + W(i + 1, j, k - 1)) : wc;
    double wB = (j > 1) ? 0.5 * (W(i, j - 1, k) + W(i, j - 1, k - 1)) : wc;
    double wT = (j < ny) ? 0.5 * (W(i, j + 1, k) + W(i, j + 1, k - 1)) : wc;
    double wF = (k > 1) ? 0.5 * (W(i, j, k - 1) + W(i, j, k - 2)) : wc;
    double wK = (k < nz) ? 0.5 * (W(i, j, k + 1) + W(i, j, k)) : wc;
    vz[m] = mu * ((wL + wR - 2 * wc) * idx2 + (wB + wT - 2 * wc) * idy2 + (wF + wK - 2 * wc) * idz2);
#undef U
#undef V
#undef W
}

void lfm_compute_viscous(CudaLFMState3D& s, CudaVel3D vel, double mu) {
    viscous_kernel<<<grid3(s.nx, s.ny, s.nz), block3()>>>(vel.u, vel.v, vel.w, s.g_.solid, s.visc_x,
                                                          s.visc_y, s.visc_z, geom(s), mu);
}

// u0 += coeff * (cell-centered field cx/cy/cz averaged to faces).
__global__ void accum_u_kernel(double* u0u, const double* cx, const bool* solid, double coeff,
                               GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i >= G.nx || j > G.ny || k > G.nz)
        return;
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)] || solid[lfm_ip(i + 1, j, k, G.nx, G.ny)])
        return;
    u0u[lfm_iu(i, j, k, G.nx, G.ny)] +=
        coeff * 0.5 * (cx[lfm_fm(i, j, k, G.nx, G.ny)] + cx[lfm_fm(i + 1, j, k, G.nx, G.ny)]);
}
__global__ void accum_v_kernel(double* u0v, const double* cy, const bool* solid, double coeff,
                               GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j >= G.ny || k > G.nz)
        return;
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)] || solid[lfm_ip(i, j + 1, k, G.nx, G.ny)])
        return;
    u0v[lfm_iv(i, j, k, G.nx, G.ny)] +=
        coeff * 0.5 * (cy[lfm_fm(i, j, k, G.nx, G.ny)] + cy[lfm_fm(i, j + 1, k, G.nx, G.ny)]);
}
__global__ void accum_w_kernel(double* u0w, const double* cz, const bool* solid, double coeff,
                               GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k >= G.nz)
        return;
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)] || solid[lfm_ip(i, j, k + 1, G.nx, G.ny)])
        return;
    u0w[lfm_iw(i, j, k, G.nx, G.ny)] +=
        coeff * 0.5 * (cz[lfm_fm(i, j, k, G.nx, G.ny)] + cz[lfm_fm(i, j, k + 1, G.nx, G.ny)]);
}

void lfm_accumulate_to_u0(CudaLFMState3D& s, CudaVel3D u0, const double* cx, const double* cy,
                          const double* cz, double coeff) {
    dim3 blk = block3(), gr = grid3(s.nx, s.ny, s.nz);
    accum_u_kernel<<<gr, blk>>>(u0.u, cx, s.g_.solid, coeff, geom(s));
    accum_v_kernel<<<gr, blk>>>(u0.v, cy, s.g_.solid, coeff, geom(s));
    accum_w_kernel<<<gr, blk>>>(u0.w, cz, s.g_.solid, coeff, geom(s));
}

// Path integral cell-centered contribution: cc = F_mid^T · visc(Φ_mid).
__global__ void path_integral_cc_kernel(const double* vx, const double* vy, const double* vz,
                                        const double* pmx, const double* pmy, const double* pmz,
                                        const double* Fm0, const double* Fm1, const double* Fm2,
                                        const double* Fm3, const double* Fm4, const double* Fm5,
                                        const double* Fm6, const double* Fm7, const double* Fm8,
                                        const bool* solid, double* ox, double* oy, double* oz,
                                        GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    long m = lfm_fm(i, j, k, G.nx, G.ny);
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)]) {
        ox[m] = oy[m] = oz[m] = 0.0;
        return;
    }
    double sx, sy, sz;
    d_sample_cell_centered(vx, vy, vz, pmx[m], pmy[m], pmz[m], G.nx, G.ny, G.nz, G.dx, G.dy, G.dz,
                           G.Lx, G.Ly, G.Lz, sx, sy, sz);
    ox[m] = Fm0[m] * sx + Fm3[m] * sy + Fm6[m] * sz;
    oy[m] = Fm1[m] * sx + Fm4[m] * sy + Fm7[m] * sz;
    oz[m] = Fm2[m] * sx + Fm5[m] * sy + Fm8[m] * sz;
}

void lfm_accumulate_path_integral(CudaLFMState3D& s, CudaVel3D u0, double coeff) {
    // F_mid^T · visc(Φ_mid) → uhat (scratch), then average to faces of u0.
    path_integral_cc_kernel<<<grid3(s.nx, s.ny, s.nz), block3()>>>(
        s.visc_x, s.visc_y, s.visc_z, s.phi_mid_x, s.phi_mid_y, s.phi_mid_z, s.F_mid[0], s.F_mid[1],
        s.F_mid[2], s.F_mid[3], s.F_mid[4], s.F_mid[5], s.F_mid[6], s.F_mid[7], s.F_mid[8],
        s.g_.solid, s.uhat_x, s.uhat_y, s.uhat_z, geom(s));
    lfm_accumulate_to_u0(s, u0, s.uhat_x, s.uhat_y, s.uhat_z, coeff);
}

void lfm_save_flow_map_state(CudaLFMState3D& s) {
    long fs   = lfm_fm_size(s.nx, s.ny, s.nz);
    size_t nb = (size_t)fs * sizeof(double);
    cudaMemcpy(s.phi_mid_x, s.phi_x, nb, cudaMemcpyDeviceToDevice);
    cudaMemcpy(s.phi_mid_y, s.phi_y, nb, cudaMemcpyDeviceToDevice);
    cudaMemcpy(s.phi_mid_z, s.phi_z, nb, cudaMemcpyDeviceToDevice);
    for (int a = 0; a < 9; a++)
        cudaMemcpy(s.F_mid[a], s.F[a], nb, cudaMemcpyDeviceToDevice);
}

__global__ void midpoints_kernel(double* pmx, double* pmy, double* pmz, const double* px,
                                 const double* py, const double* pz, double* Fm0, double* Fm1,
                                 double* Fm2, double* Fm3, double* Fm4, double* Fm5, double* Fm6,
                                 double* Fm7, double* Fm8, const double* F0, const double* F1,
                                 const double* F2, const double* F3, const double* F4,
                                 const double* F5, const double* F6, const double* F7,
                                 const double* F8, long N) {
    long m = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N)
        return;
    pmx[m] = 0.5 * (pmx[m] + px[m]);
    pmy[m] = 0.5 * (pmy[m] + py[m]);
    pmz[m] = 0.5 * (pmz[m] + pz[m]);
    Fm0[m] = 0.5 * (Fm0[m] + F0[m]);
    Fm1[m] = 0.5 * (Fm1[m] + F1[m]);
    Fm2[m] = 0.5 * (Fm2[m] + F2[m]);
    Fm3[m] = 0.5 * (Fm3[m] + F3[m]);
    Fm4[m] = 0.5 * (Fm4[m] + F4[m]);
    Fm5[m] = 0.5 * (Fm5[m] + F5[m]);
    Fm6[m] = 0.5 * (Fm6[m] + F6[m]);
    Fm7[m] = 0.5 * (Fm7[m] + F7[m]);
    Fm8[m] = 0.5 * (Fm8[m] + F8[m]);
}

void lfm_compute_midpoints(CudaLFMState3D& s) {
    long N = lfm_fm_size(s.nx, s.ny, s.nz);
    midpoints_kernel<<<(int)((N + 255) / 256), 256>>>(
        s.phi_mid_x, s.phi_mid_y, s.phi_mid_z, s.phi_x, s.phi_y, s.phi_z, s.F_mid[0], s.F_mid[1],
        s.F_mid[2], s.F_mid[3], s.F_mid[4], s.F_mid[5], s.F_mid[6], s.F_mid[7], s.F_mid[8], s.F[0],
        s.F[1], s.F[2], s.F[3], s.F[4], s.F[5], s.F[6], s.F[7], s.F[8], N);
}

// Pullback: m = T^T · u0(Ψ).
__global__ void pullback_kernel(const double* u0u, const double* u0v, const double* u0w,
                                const double* psix, const double* psiy, const double* psiz,
                                const double* T0, const double* T1, const double* T2,
                                const double* T3, const double* T4, const double* T5,
                                const double* T6, const double* T7, const double* T8, double* mx,
                                double* my, double* mz, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    long m = lfm_fm(i, j, k, G.nx, G.ny);
    double uX, uY, uZ;
    d_sample_velocity(u0u, u0v, u0w, psix[m], psiy[m], psiz[m], G.nx, G.ny, G.nz, G.dx, G.dy, G.dz,
                      G.Lx, G.Ly, G.Lz, uX, uY, uZ);
    mx[m] = T0[m] * uX + T3[m] * uY + T6[m] * uZ;
    my[m] = T1[m] * uX + T4[m] * uY + T7[m] * uZ;
    mz[m] = T2[m] * uX + T5[m] * uY + T8[m] * uZ;
}

void lfm_pullback_impulse(CudaLFMState3D& s, CudaVel3D u0) {
    pullback_kernel<<<grid3(s.nx, s.ny, s.nz), block3()>>>(
        u0.u, u0.v, u0.w, s.psi_x, s.psi_y, s.psi_z, s.T[0], s.T[1], s.T[2], s.T[3], s.T[4], s.T[5],
        s.T[6], s.T[7], s.T[8], s.m_x, s.m_y, s.m_z, geom(s));
}

// Forward pullback: û = F^T · m(Φ).
__global__ void forward_pullback_kernel(const double* mx, const double* my, const double* mz,
                                        const double* px, const double* py, const double* pz,
                                        const double* F0, const double* F1, const double* F2,
                                        const double* F3, const double* F4, const double* F5,
                                        const double* F6, const double* F7, const double* F8,
                                        double* ox, double* oy, double* oz, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    long m = lfm_fm(i, j, k, G.nx, G.ny);
    double sx, sy, sz;
    d_sample_cell_centered(mx, my, mz, px[m], py[m], pz[m], G.nx, G.ny, G.nz, G.dx, G.dy, G.dz, G.Lx,
                           G.Ly, G.Lz, sx, sy, sz);
    ox[m] = F0[m] * sx + F3[m] * sy + F6[m] * sz;
    oy[m] = F1[m] * sx + F4[m] * sy + F7[m] * sz;
    oz[m] = F2[m] * sx + F5[m] * sy + F8[m] * sz;
}

void lfm_forward_pullback(CudaLFMState3D& s, const double* mx, const double* my, const double* mz,
                          double* ox, double* oy, double* oz) {
    forward_pullback_kernel<<<grid3(s.nx, s.ny, s.nz), block3()>>>(
        mx, my, mz, s.phi_x, s.phi_y, s.phi_z, s.F[0], s.F[1], s.F[2], s.F[3], s.F[4], s.F[5],
        s.F[6], s.F[7], s.F[8], ox, oy, oz, geom(s));
}

// Error correction (Steps 23-26): e = (F^T m(Φ) − u0_cc)·0.5, then m −= T^T e(Ψ).
__global__ void error_e_kernel(const double* uhx, const double* uhy, const double* uhz,
                               const double* u0u, const double* u0v, const double* u0w, double* ex,
                               double* ey, double* ez, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    long m     = lfm_fm(i, j, k, G.nx, G.ny);
    double uc0 = 0.5 * (u0u[lfm_iu(i, j, k, G.nx, G.ny)] + u0u[lfm_iu(i - 1, j, k, G.nx, G.ny)]);
    double vc0 = 0.5 * (u0v[lfm_iv(i, j, k, G.nx, G.ny)] + u0v[lfm_iv(i, j - 1, k, G.nx, G.ny)]);
    double wc0 = 0.5 * (u0w[lfm_iw(i, j, k, G.nx, G.ny)] + u0w[lfm_iw(i, j, k - 1, G.nx, G.ny)]);
    ex[m]      = (uhx[m] - uc0) * 0.5;
    ey[m]      = (uhy[m] - vc0) * 0.5;
    ez[m]      = (uhz[m] - wc0) * 0.5;
}
__global__ void error_sub_kernel(const double* ex, const double* ey, const double* ez,
                                 const double* psix, const double* psiy, const double* psiz,
                                 const double* T0, const double* T1, const double* T2,
                                 const double* T3, const double* T4, const double* T5,
                                 const double* T6, const double* T7, const double* T8, double* mx,
                                 double* my, double* mz, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    long m = lfm_fm(i, j, k, G.nx, G.ny);
    double esx, esy, esz;
    d_sample_cell_centered(ex, ey, ez, psix[m], psiy[m], psiz[m], G.nx, G.ny, G.nz, G.dx, G.dy, G.dz,
                           G.Lx, G.Ly, G.Lz, esx, esy, esz);
    mx[m] -= T0[m] * esx + T3[m] * esy + T6[m] * esz;
    my[m] -= T1[m] * esx + T4[m] * esy + T7[m] * esz;
    mz[m] -= T2[m] * esx + T5[m] * esy + T8[m] * esz;
}

// BFECC clamp (paper's BfeccClampKernel): bound each corrected impulse
// component to the [min,max] of its 6 face-neighbours' un-corrected (pre)
// values. Center excluded — mirrors the reference and the CPU golden path.
__device__ inline void d_clamp_one(double* m, const double* pre, int i, int j, int k, GeomParams G) {
    const int di[6] = {-1, 1, 0, 0, 0, 0};
    const int dj[6] = {0, 0, -1, 1, 0, 0};
    const int dk[6] = {0, 0, 0, 0, -1, 1};
    double lo = 0.0, hi = 0.0;
    bool first = true;
    for (int n = 0; n < 6; n++) {
        int a = i + di[n], b = j + dj[n], c = k + dk[n];
        if (a < 1 || a > G.nx || b < 1 || b > G.ny || c < 1 || c > G.nz)
            continue;
        double v = pre[lfm_fm(a, b, c, G.nx, G.ny)];
        if (first) {
            lo = hi = v;
            first   = false;
        } else {
            lo = fmin(lo, v);
            hi = fmax(hi, v);
        }
    }
    if (first)
        return;
    long id = lfm_fm(i, j, k, G.nx, G.ny);
    if (m[id] < lo)
        m[id] = lo;
    else if (m[id] > hi)
        m[id] = hi;
}
__global__ void bfecc_clamp_kernel(double* mx, double* my, double* mz, const double* px,
                                   const double* py, const double* pz, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k > G.nz)
        return;
    d_clamp_one(mx, px, i, j, k, G);
    d_clamp_one(my, py, i, j, k, G);
    d_clamp_one(mz, pz, i, j, k, G);
}

void lfm_error_correction(CudaLFMState3D& s, CudaVel3D u0, bool clamp) {
    lfm_forward_pullback(s, s.m_x, s.m_y, s.m_z, s.uhat_x, s.uhat_y, s.uhat_z);
    dim3 gr = grid3(s.nx, s.ny, s.nz), blk = block3();
    error_e_kernel<<<gr, blk>>>(s.uhat_x, s.uhat_y, s.uhat_z, u0.u, u0.v, u0.w, s.e_x, s.e_y, s.e_z,
                                geom(s));
    // Save the un-corrected impulse into the now-free uhat buffers (paper's `u`).
    size_t fs = sizeof(double) * (size_t)s.nx * s.ny * s.nz;
    if (clamp) {
        cudaMemcpy(s.uhat_x, s.m_x, fs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(s.uhat_y, s.m_y, fs, cudaMemcpyDeviceToDevice);
        cudaMemcpy(s.uhat_z, s.m_z, fs, cudaMemcpyDeviceToDevice);
    }
    error_sub_kernel<<<gr, blk>>>(s.e_x, s.e_y, s.e_z, s.psi_x, s.psi_y, s.psi_z, s.T[0], s.T[1],
                                  s.T[2], s.T[3], s.T[4], s.T[5], s.T[6], s.T[7], s.T[8], s.m_x,
                                  s.m_y, s.m_z, geom(s));
    if (clamp)
        bfecc_clamp_kernel<<<gr, blk>>>(s.m_x, s.m_y, s.m_z, s.uhat_x, s.uhat_y, s.uhat_z, geom(s));
}

// Gauge write-back (Step 27): u_n ← face-average of the impulse m.
__global__ void gauge_u_kernel(double* u, const double* mx, const bool* solid, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i >= G.nx || j > G.ny || k > G.nz)
        return;
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)] || solid[lfm_ip(i + 1, j, k, G.nx, G.ny)])
        return;
    u[lfm_iu(i, j, k, G.nx, G.ny)] =
        0.5 * (mx[lfm_fm(i, j, k, G.nx, G.ny)] + mx[lfm_fm(i + 1, j, k, G.nx, G.ny)]);
}
__global__ void gauge_v_kernel(double* v, const double* my, const bool* solid, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j >= G.ny || k > G.nz)
        return;
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)] || solid[lfm_ip(i, j + 1, k, G.nx, G.ny)])
        return;
    v[lfm_iv(i, j, k, G.nx, G.ny)] =
        0.5 * (my[lfm_fm(i, j, k, G.nx, G.ny)] + my[lfm_fm(i, j + 1, k, G.nx, G.ny)]);
}
__global__ void gauge_w_kernel(double* w, const double* mz, const bool* solid, GeomParams G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > G.nx || j > G.ny || k >= G.nz)
        return;
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)] || solid[lfm_ip(i, j, k + 1, G.nx, G.ny)])
        return;
    w[lfm_iw(i, j, k, G.nx, G.ny)] =
        0.5 * (mz[lfm_fm(i, j, k, G.nx, G.ny)] + mz[lfm_fm(i, j, k + 1, G.nx, G.ny)]);
}

void lfm_gauge_writeback(CudaLFMState3D& s, CudaVel3D vel) {
    dim3 gr = grid3(s.nx, s.ny, s.nz), blk = block3();
    gauge_u_kernel<<<gr, blk>>>(vel.u, s.m_x, s.g_.solid, geom(s));
    gauge_v_kernel<<<gr, blk>>>(vel.v, s.m_y, s.g_.solid, geom(s));
    gauge_w_kernel<<<gr, blk>>>(vel.w, s.m_z, s.g_.solid, geom(s));
}

void lfm_copy_vel(CudaLFMState3D& s, CudaVel3D dst, CudaVel3D src) {
    cudaMemcpy(dst.u, src.u, (size_t)lfm_u_size(s.nx, s.ny, s.nz) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst.v, src.v, (size_t)lfm_v_size(s.nx, s.ny, s.nz) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(dst.w, src.w, (size_t)lfm_w_size(s.nx, s.ny, s.nz) * sizeof(double),
               cudaMemcpyDeviceToDevice);
}

// ══════════════════════════════════════════════════════════════════════
// FIX① — per-axis (staggered-face) flow maps (GPU). Device port of the CPU
// FaceFlowMap routines: per-face identity init, RK4 (pos+covector) march,
// per-face pullback, and the face BFECC error correction. The impulse rides
// directly on the MAC faces (zero center↔face averaging). Arithmetic mirrors
// the CPU face_* functions bit-for-bit (FMA off → GPU==CPU, gated by P2-P4).
// ══════════════════════════════════════════════════════════════════════

// One face flow-map's device pointers for one axis (mirror of CudaFaceFlowMap).
struct FaceMapPtrs {
    double *bx, *by, *bz, *t0, *t1, *t2;
    double *fx, *fy, *fz, *f0, *f1, *f2;
};

// Per-axis geometry: index range [1..imax]x[1..jmax]x[1..kmax] and MAC index.
// axis: 0=u,1=v,2=w. Returns the flat MAC index for (i,j,k).
__device__ inline int d_face_idx(int axis, int i, int j, int k, int nx, int ny) {
    return axis == 0 ? lfm_iu(i, j, k, nx, ny)
                     : (axis == 1 ? lfm_iv(i, j, k, nx, ny) : lfm_iw(i, j, k, nx, ny));
}
__device__ inline void d_face_coord(int axis, int i, int j, int k, double dx, double dy, double dz,
                                    double& x, double& y, double& z) {
    if (axis == 0) {
        x = i * dx;
        y = (j - 0.5) * dy;
        z = (k - 0.5) * dz;
    } else if (axis == 1) {
        x = (i - 0.5) * dx;
        y = j * dy;
        z = (k - 0.5) * dz;
    } else {
        x = (i - 0.5) * dx;
        y = (j - 0.5) * dy;
        z = k * dz;
    }
}

// Per-axis face index limits for axis a (interior faces only, matching CPU).
__host__ __device__ inline void d_face_limits(int axis, int nx, int ny, int nz, int& imax,
                                              int& jmax, int& kmax) {
    imax = (axis == 0) ? nx - 1 : nx;
    jmax = (axis == 1) ? ny - 1 : ny;
    kmax = (axis == 2) ? nz - 1 : nz;
}

__global__ void face_identity_kernel(FaceMapPtrs f, int axis, bool fwd, GeomParams G) {
    int imax, jmax, kmax;
    d_face_limits(axis, G.nx, G.ny, G.nz, imax, jmax, kmax);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > imax || j > jmax || k > kmax)
        return;
    int m = d_face_idx(axis, i, j, k, G.nx, G.ny);
    double x, y, z;
    d_face_coord(axis, i, j, k, G.dx, G.dy, G.dz, x, y, z);
    double e0 = (axis == 0) ? 1.0 : 0.0, e1 = (axis == 1) ? 1.0 : 0.0,
           e2 = (axis == 2) ? 1.0 : 0.0;
    if (fwd) {
        f.fx[m] = x;
        f.fy[m] = y;
        f.fz[m] = z;
        f.f0[m] = e0;
        f.f1[m] = e1;
        f.f2[m] = e2;
    } else {
        f.bx[m] = x;
        f.by[m] = y;
        f.bz[m] = z;
        f.t0[m] = e0;
        f.t1[m] = e1;
        f.t2[m] = e2;
    }
}

// RK4 of one face point: state s[6]=[pos(3),covector(3)]. Mirrors the CPU
// face_march_point (non-incremental k1..k4). T=double reproduces the CPU bit-
// for-bit; T=float is the fast production path (author runs FP32 throughout).
template <typename T, typename VT>
__device__ inline void d_face_march_point(T& px, T& py, T& pz, T Tcov[3],
                                          const VT* uu, const VT* vv, const VT* ww,
                                          T dt_march, GeomParams G) {
    T Gdx = (T)G.dx, Gdy = (T)G.dy, Gdz = (T)G.dz, GLx = (T)G.Lx, GLy = (T)G.Ly, GLz = (T)G.Lz;
    T s0[6] = {px, py, pz, Tcov[0], Tcov[1], Tcov[2]};
    auto rhs = [&](const T s[6], T d[6]) {
        T vu, vvel, vw, g[9];
        d_sample_velocity_grad<T, VT>(uu, vv, ww, s[0], s[1], s[2], G.nx, G.ny, G.nz, Gdx, Gdy, Gdz, GLx,
                                  GLy, GLz, vu, vvel, vw, g);
        d[0] = vu;
        d[1] = vvel;
        d[2] = vw;
        for (int a = 0; a < 3; a++)
            d[3 + a] = g[3 * a + 0] * s[3 + 0] + g[3 * a + 1] * s[3 + 1] + g[3 * a + 2] * s[3 + 2];
    };
    T k1[6], k2[6], k3[6], k4[6], tmp[6], out[6];
    rhs(s0, k1);
    for (int m = 0; m < 6; m++) {
        k1[m] *= dt_march;
        tmp[m] = s0[m] + T(0.5) * k1[m];
    }
    rhs(tmp, k2);
    for (int m = 0; m < 6; m++) {
        k2[m] *= dt_march;
        tmp[m] = s0[m] + T(0.5) * k2[m];
    }
    rhs(tmp, k3);
    for (int m = 0; m < 6; m++) {
        k3[m] *= dt_march;
        tmp[m] = s0[m] + k3[m];
    }
    rhs(tmp, k4);
    for (int m = 0; m < 6; m++) {
        k4[m] *= dt_march;
        out[m] = s0[m] + (k1[m] + T(2) * k2[m] + T(2) * k3[m] + k4[m]) / T(6.0);
    }
    px = tclamp<T>(out[0], T(0), GLx);
    py = tclamp<T>(out[1], T(0), GLy);
    pz = tclamp<T>(out[2], T(0), GLz);
    for (int a = 0; a < 3; a++) {
        T val = out[3 + a];
        if (!isfinite(val))
            val = T(0);
        Tcov[a] = val;
    }
}

template <typename T, typename VT>
__global__ void __launch_bounds__(256, 2) face_march_kernel(FaceMapPtrs f, int axis, bool fwd, const VT* uu,
                                  const VT* vv, const VT* ww, double dt_march,
                                  GeomParams G) {
    int imax, jmax, kmax;
    d_face_limits(axis, G.nx, G.ny, G.nz, imax, jmax, kmax);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > imax || j > jmax || k > kmax)
        return;
    int m = d_face_idx(axis, i, j, k, G.nx, G.ny);
    T dtm = (T)dt_march;
    if (fwd) {
        T Tcov[3] = {(T)f.f0[m], (T)f.f1[m], (T)f.f2[m]};
        T px = (T)f.fx[m], py = (T)f.fy[m], pz = (T)f.fz[m];
        d_face_march_point<T, VT>(px, py, pz, Tcov, uu, vv, ww, dtm, G);
        f.fx[m] = px;
        f.fy[m] = py;
        f.fz[m] = pz;
        f.f0[m] = Tcov[0];
        f.f1[m] = Tcov[1];
        f.f2[m] = Tcov[2];
    } else {
        T Tcov[3] = {(T)f.t0[m], (T)f.t1[m], (T)f.t2[m]};
        T px = (T)f.bx[m], py = (T)f.by[m], pz = (T)f.bz[m];
        d_face_march_point<T, VT>(px, py, pz, Tcov, uu, vv, ww, dtm, G);
        f.bx[m] = px;
        f.by[m] = py;
        f.bz[m] = pz;
        f.t0[m] = Tcov[0];
        f.t1[m] = Tcov[1];
        f.t2[m] = Tcov[2];
    }
}

// Per-face pullback dst_a = T_a · src(ψ_a) (fwd=false: backward map; fwd=true: forward).
template <typename T, typename VT>
__global__ void face_pullback_kernel(FaceMapPtrs f, int axis, bool fwd, const VT* su,
                                     const VT* sv, const VT* sw, double* dst, GeomParams G) {
    int imax, jmax, kmax;
    d_face_limits(axis, G.nx, G.ny, G.nz, imax, jmax, kmax);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > imax || j > jmax || k > kmax)
        return;
    int m = d_face_idx(axis, i, j, k, G.nx, G.ny);
    T px = (T)(fwd ? f.fx[m] : f.bx[m]);
    T py = (T)(fwd ? f.fy[m] : f.by[m]);
    T pz = (T)(fwd ? f.fz[m] : f.bz[m]);
    T T0 = (T)(fwd ? f.f0[m] : f.t0[m]);
    T T1 = (T)(fwd ? f.f1[m] : f.t1[m]);
    T T2 = (T)(fwd ? f.f2[m] : f.t2[m]);
    T sX, sY, sZ;
    d_sample_velocity<T, VT>(su, sv, sw, px, py, pz, G.nx, G.ny, G.nz, (T)G.dx, (T)G.dy, (T)G.dz, (T)G.Lx, (T)G.Ly, (T)G.Lz,
                         sX, sY, sZ);
    dst[m] = T0 * sX + T1 * sY + T2 * sZ;
}

static FaceMapPtrs face_ptrs(CudaLFMState3D::CudaFaceFlowMap& f) {
    FaceMapPtrs p;
    p.bx = f.bx;
    p.by = f.by;
    p.bz = f.bz;
    p.t0 = f.t0;
    p.t1 = f.t1;
    p.t2 = f.t2;
    p.fx = f.fx;
    p.fy = f.fy;
    p.fz = f.fz;
    p.f0 = f.f0;
    p.f1 = f.f1;
    p.f2 = f.f2;
    return p;
}

static dim3 grid_face(int imax, int jmax, int kmax) {
    return dim3((imax + 7) / 8, (jmax + 7) / 8, (kmax + 7) / 8);
}

void lfm_face_set_forward_identity(CudaLFMState3D& s) {
    GeomParams G = geom(s);
    int im, jm, km;
    for (int axis = 0; axis < 3; axis++) {
        d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
        CudaLFMState3D::CudaFaceFlowMap& f = (axis == 0) ? s.fmu : (axis == 1 ? s.fmv : s.fmw);
        face_identity_kernel<<<grid_face(im, jm, km), block3()>>>(face_ptrs(f), axis, true, G);
    }
}
void lfm_face_set_backward_identity(CudaLFMState3D& s) {
    GeomParams G = geom(s);
    int im, jm, km;
    for (int axis = 0; axis < 3; axis++) {
        d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
        CudaLFMState3D::CudaFaceFlowMap& f = (axis == 0) ? s.fmu : (axis == 1 ? s.fmv : s.fmw);
        face_identity_kernel<<<grid_face(im, jm, km), block3()>>>(face_ptrs(f), axis, false, G);
    }
}
// The 6-state RK4 (pos+covector) is register/local-memory heavy; a 512-thread
// block overflows resources. Use 256 threads (8×8×4) like the cell march.
// Launch the per-axis face march. In the fp32 path the velocity field is staged
// once into FP32 scratch (s.vu32/…) so the hot sampling gather is pure float and
// avoids the per-load FP64 convert (the prior #1 hotspot). The double (test) path
// reads the double arrays directly and stays bit-exact vs CPU.
static void face_march_run(CudaLFMState3D& s, CudaVel3D vel, double dt_march, bool fwd) {
    GeomParams G = geom(s);
    int im, jm, km;
    dim3 blk(8, 8, 4);
    bool fp16 = fp16_sampling(s);
    if (fp16)
        convert_vel_f16(vel, (__half*)s.vu16, (__half*)s.vv16, (__half*)s.vw16,
                        lfm_u_size(s.nx, s.ny, s.nz), lfm_v_size(s.nx, s.ny, s.nz),
                        lfm_w_size(s.nx, s.ny, s.nz));
    else if (s.fp32_march)
        convert_vel_f32(vel, s.vu32, s.vv32, s.vw32, lfm_u_size(s.nx, s.ny, s.nz),
                        lfm_v_size(s.nx, s.ny, s.nz), lfm_w_size(s.nx, s.ny, s.nz));
    for (int axis = 0; axis < 3; axis++) {
        d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
        CudaLFMState3D::CudaFaceFlowMap& f = (axis == 0) ? s.fmu : (axis == 1 ? s.fmv : s.fmw);
        dim3 gr((im + 7) / 8, (jm + 7) / 8, (km + 3) / 4);
        if (fp16)
            face_march_kernel<float, __half><<<gr, blk>>>(face_ptrs(f), axis, fwd, (__half*)s.vu16, (__half*)s.vv16, (__half*)s.vw16, dt_march, G);
        else if (s.fp32_march)
            face_march_kernel<float, float><<<gr, blk>>>(face_ptrs(f), axis, fwd, s.vu32, s.vv32, s.vw32, dt_march, G);
        else
            face_march_kernel<double, double><<<gr, blk>>>(face_ptrs(f), axis, fwd, vel.u, vel.v, vel.w, dt_march, G);
    }
}
void lfm_face_march_forward(CudaLFMState3D& s, CudaVel3D vel, double dt_march) {
    face_march_run(s, vel, dt_march, true);
}
void lfm_face_march_backward(CudaLFMState3D& s, CudaVel3D vel, double dt_march) {
    face_march_run(s, vel, dt_march, false);
}
void lfm_face_pullback(CudaLFMState3D& s, CudaVel3D src, CudaVel3D dst, bool fwd) {
    GeomParams G = geom(s);
    int im, jm, km;
    bool fp16 = fp16_sampling(s);
    if (fp16)
        convert_vel_f16(src, (__half*)s.su16, (__half*)s.sv16, (__half*)s.sw16,
                        lfm_u_size(s.nx, s.ny, s.nz), lfm_v_size(s.nx, s.ny, s.nz),
                        lfm_w_size(s.nx, s.ny, s.nz));
    else if (s.fp32_march)
        convert_vel_f32(src, s.su32, s.sv32, s.sw32, lfm_u_size(s.nx, s.ny, s.nz),
                        lfm_v_size(s.nx, s.ny, s.nz), lfm_w_size(s.nx, s.ny, s.nz));
    for (int axis = 0; axis < 3; axis++) {
        d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
        CudaLFMState3D::CudaFaceFlowMap& f = (axis == 0) ? s.fmu : (axis == 1 ? s.fmv : s.fmw);
        double* d = (axis == 0) ? dst.u : (axis == 1 ? dst.v : dst.w);
        if (fp16)
            face_pullback_kernel<float, __half><<<grid_face(im, jm, km), block3()>>>(face_ptrs(f), axis, fwd, (__half*)s.su16, (__half*)s.sv16, (__half*)s.sw16, d, G);
        else if (s.fp32_march)
            face_pullback_kernel<float, float><<<grid_face(im, jm, km), block3()>>>(face_ptrs(f), axis, fwd, s.su32, s.sv32, s.sw32, d, G);
        else
            face_pullback_kernel<double, double><<<grid_face(im, jm, km), block3()>>>(face_ptrs(f), axis, fwd, src.u, src.v, src.w, d, G);
    }
}

// e_a = û0_a − u0_a (per-face, on the same MAC grid). Only interior faces touched.
__global__ void face_err_kernel(int axis, const double* uh, const double* u0, double* e,
                                GeomParams G) {
    int imax, jmax, kmax;
    d_face_limits(axis, G.nx, G.ny, G.nz, imax, jmax, kmax);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > imax || j > jmax || k > kmax)
        return;
    int m = d_face_idx(axis, i, j, k, G.nx, G.ny);
    e[m]  = uh[m] - u0[m];
}

// m_a -= 0.5·corr_a (per-face).
__global__ void face_subcorr_kernel(int axis, double* m, const double* corr, GeomParams G) {
    int imax, jmax, kmax;
    d_face_limits(axis, G.nx, G.ny, G.nz, imax, jmax, kmax);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > imax || j > jmax || k > kmax)
        return;
    int id = d_face_idx(axis, i, j, k, G.nx, G.ny);
    m[id] -= 0.5 * corr[id];
}

// Clamp m_a to the [min,max] of its 6 same-face-grid neighbours' pre values.
__global__ void face_clamp_kernel(int axis, double* m, const double* pre, GeomParams G) {
    int imax, jmax, kmax;
    d_face_limits(axis, G.nx, G.ny, G.nz, imax, jmax, kmax);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > imax || j > jmax || k > kmax)
        return;
    const int di[6] = {-1, 1, 0, 0, 0, 0};
    const int dj[6] = {0, 0, -1, 1, 0, 0};
    const int dk[6] = {0, 0, 0, 0, -1, 1};
    double lo = 0.0, hi = 0.0;
    bool first = true;
    for (int n = 0; n < 6; n++) {
        int a = i + di[n], b = j + dj[n], c = k + dk[n];
        if (a < 1 || a > imax || b < 1 || b > jmax || c < 1 || c > kmax)
            continue;
        double v = pre[d_face_idx(axis, a, b, c, G.nx, G.ny)];
        if (first) {
            lo = hi = v;
            first   = false;
        } else {
            lo = fmin(lo, v);
            hi = fmax(hi, v);
        }
    }
    if (first)
        return;
    int id = d_face_idx(axis, i, j, k, G.nx, G.ny);
    if (m[id] < lo)
        m[id] = lo;
    else if (m[id] > hi)
        m[id] = hi;
}

// Write m_a → vel_a directly (impulse IS velocity), skipping solid-adjacent faces.
__global__ void face_write_vel_kernel(int axis, double* vel, const double* m, const bool* solid,
                                      GeomParams G) {
    int imax, jmax, kmax;
    d_face_limits(axis, G.nx, G.ny, G.nz, imax, jmax, kmax);
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i > imax || j > jmax || k > kmax)
        return;
    int sp1 = (axis == 0) ? lfm_ip(i + 1, j, k, G.nx, G.ny)
                          : (axis == 1 ? lfm_ip(i, j + 1, k, G.nx, G.ny)
                                       : lfm_ip(i, j, k + 1, G.nx, G.ny));
    if (solid[lfm_ip(i, j, k, G.nx, G.ny)] || solid[sp1])
        return;
    int id  = d_face_idx(axis, i, j, k, G.nx, G.ny);
    vel[id] = m[id];
}

void lfm_face_error_correction(CudaLFMState3D& s, CudaVel3D u0, CudaVel3D vel, bool clamp) {
    GeomParams G = geom(s);
    int im, jm, km;
    // û0 = F·m(φ) → mhat.
    lfm_face_pullback(s, s.mface, s.mhat, /*fwd=*/true);
    // e = û0 − u0 (per axis) → merr.
    for (int axis = 0; axis < 3; axis++) {
        d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
        const double* uh = (axis == 0) ? s.mhat.u : (axis == 1 ? s.mhat.v : s.mhat.w);
        const double* u0a = (axis == 0) ? u0.u : (axis == 1 ? u0.v : u0.w);
        double* e         = (axis == 0) ? s.merr.u : (axis == 1 ? s.merr.v : s.merr.w);
        face_err_kernel<<<grid_face(im, jm, km), block3()>>>(axis, uh, u0a, e, G);
    }
    // corr = T·e(ψ) → mhat (reuse as correction scratch).
    lfm_face_pullback(s, s.merr, s.mhat, /*fwd=*/false);
    // Save pre-correction impulse into merr for the clamp.
    if (clamp)
        lfm_copy_vel(s, s.merr, s.mface);
    // m -= 0.5·corr.
    for (int axis = 0; axis < 3; axis++) {
        d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
        double* m         = (axis == 0) ? s.mface.u : (axis == 1 ? s.mface.v : s.mface.w);
        const double* cor = (axis == 0) ? s.mhat.u : (axis == 1 ? s.mhat.v : s.mhat.w);
        face_subcorr_kernel<<<grid_face(im, jm, km), block3()>>>(axis, m, cor, G);
    }
    if (clamp) {
        for (int axis = 0; axis < 3; axis++) {
            d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
            double* m         = (axis == 0) ? s.mface.u : (axis == 1 ? s.mface.v : s.mface.w);
            const double* pre = (axis == 0) ? s.merr.u : (axis == 1 ? s.merr.v : s.merr.w);
            face_clamp_kernel<<<grid_face(im, jm, km), block3()>>>(axis, m, pre, G);
        }
    }
    // Write m → vel directly (no gauge averaging).
    for (int axis = 0; axis < 3; axis++) {
        d_face_limits(axis, s.nx, s.ny, s.nz, im, jm, km);
        double* v       = (axis == 0) ? vel.u : (axis == 1 ? vel.v : vel.w);
        const double* m = (axis == 0) ? s.mface.u : (axis == 1 ? s.mface.v : s.mface.w);
        face_write_vel_kernel<<<grid_face(im, jm, km), block3()>>>(axis, v, m, s.g_.solid, G);
    }
}
