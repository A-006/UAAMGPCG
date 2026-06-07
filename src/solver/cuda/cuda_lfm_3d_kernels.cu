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
#include <vector>

namespace {
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
dim3 block3() { return dim3(8, 8, 8); }
dim3 grid3(int nx, int ny, int nz) {
    return dim3((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
}
} // namespace

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

// 27-point quadratic B-spline velocity interpolation (MAC-aware).
__device__ inline void d_sample_velocity(const double* uu, const double* vv, const double* ww,
                                         double x, double y, double z, int nx, int ny, int nz,
                                         double dx, double dy, double dz, double Lx, double Ly,
                                         double Lz, double& vu, double& vvel, double& vw) {
    x = dclamp(x, 0.0, Lx);
    y = dclamp(y, 0.0, Ly);
    z = dclamp(z, 0.0, Lz);

    // u-face at (i·dx,(j-0.5)·dy,(k-0.5)·dz)
    {
        double cx = x / dx, cy = y / dy + 0.5, cz = z / dz + 0.5;
        int ic = (int)floor(cx + 0.5), jc = (int)floor(cy + 0.5), kc = (int)floor(cz + 0.5);
        double wx[3], wy[3], wz[3];
        d_bspline(cx - ic, wx);
        d_bspline(cy - jc, wy);
        d_bspline(cz - kc, wz);
        double s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 0, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * uu[lfm_iu(ii, jj, kk, nx, ny)];
                }
        vu = s;
    }
    // v-face at ((i-0.5)·dx, j·dy, (k-0.5)·dz)
    {
        double cx = x / dx + 0.5, cy = y / dy, cz = z / dz + 0.5;
        int ic = (int)floor(cx + 0.5), jc = (int)floor(cy + 0.5), kc = (int)floor(cz + 0.5);
        double wx[3], wy[3], wz[3];
        d_bspline(cx - ic, wx);
        d_bspline(cy - jc, wy);
        d_bspline(cz - kc, wz);
        double s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 0, ny),
                        kk = iclamp(kc + dk, 1, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * vv[lfm_iv(ii, jj, kk, nx, ny)];
                }
        vvel = s;
    }
    // w-face at ((i-0.5)·dx,(j-0.5)·dy, k·dz)
    {
        double cx = x / dx + 0.5, cy = y / dy + 0.5, cz = z / dz;
        int ic = (int)floor(cx + 0.5), jc = (int)floor(cy + 0.5), kc = (int)floor(cz + 0.5);
        double wx[3], wy[3], wz[3];
        d_bspline(cx - ic, wx);
        d_bspline(cy - jc, wy);
        d_bspline(cz - kc, wz);
        double s = 0;
        for (int dk = -1; dk <= 1; dk++)
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int ii = iclamp(ic + di, 1, nx), jj = iclamp(jc + dj, 1, ny),
                        kk = iclamp(kc + dk, 0, nz);
                    s += wx[di + 1] * wy[dj + 1] * wz[dk + 1] * ww[lfm_iw(ii, jj, kk, nx, ny)];
                }
        vw = s;
    }
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
    s.pcg_.solve(s.g_, s.d_p, s.d_rhs, iters, tol);
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

__device__ inline void d_backtrace(const double* vu, const double* vv, const double* vw, double x,
                                    double y, double z, double dt_step, const GeomParams g,
                                    double& xo, double& yo, double& zo) {
    double u1, v1, w1;
    d_sample_velocity(vu, vv, vw, x, y, z, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.Lx, g.Ly, g.Lz, u1,
                      v1, w1);
    double xm = x - 0.5 * dt_step * u1, ym = y - 0.5 * dt_step * v1, zm = z - 0.5 * dt_step * w1;
    double um, vm, wm;
    d_sample_velocity(vu, vv, vw, xm, ym, zm, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.Lx, g.Ly, g.Lz,
                      um, vm, wm);
    xo = x - dt_step * um;
    yo = y - dt_step * vm;
    zo = z - dt_step * wm;
}

__global__ void advect_u_kernel(double* dst, const double* su, const double* sv, const double* sw,
                                 const double* vu, const double* vv, const double* vw,
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
    double xs, ys, zs, ou, ov, ow;
    d_backtrace(vu, vv, vw, i * g.dx, (j - 0.5) * g.dy, (k - 0.5) * g.dz, dt_step, g, xs, ys, zs);
    d_sample_velocity(su, sv, sw, xs, ys, zs, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.Lx, g.Ly, g.Lz,
                      ou, ov, ow);
    dst[id] = ou;
}
__global__ void advect_v_kernel(double* dst, const double* su, const double* sv, const double* sw,
                                 const double* vu, const double* vv, const double* vw,
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
    double xs, ys, zs, ou, ov, ow;
    d_backtrace(vu, vv, vw, (i - 0.5) * g.dx, j * g.dy, (k - 0.5) * g.dz, dt_step, g, xs, ys, zs);
    d_sample_velocity(su, sv, sw, xs, ys, zs, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.Lx, g.Ly, g.Lz,
                      ou, ov, ow);
    dst[id] = ov;
}
__global__ void advect_w_kernel(double* dst, const double* su, const double* sv, const double* sw,
                                 const double* vu, const double* vv, const double* vw,
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
    double xs, ys, zs, ou, ov, ow;
    d_backtrace(vu, vv, vw, (i - 0.5) * g.dx, (j - 0.5) * g.dy, k * g.dz, dt_step, g, xs, ys, zs);
    d_sample_velocity(su, sv, sw, xs, ys, zs, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.Lx, g.Ly, g.Lz,
                      ou, ov, ow);
    dst[id] = ow;
}

static GeomParams geom(const CudaLFMState3D& s) {
    return GeomParams{s.nx, s.ny, s.nz, s.dx, s.dy, s.dz, s.Lx, s.Ly, s.Lz};
}

void lfm_rk2_advect(CudaLFMState3D& s, CudaVel3D dst, CudaVel3D src, CudaVel3D vel, double dt_step) {
    GeomParams g = geom(s);
    dim3 blk = block3(), gr = grid3(s.nx, s.ny, s.nz);
    advect_u_kernel<<<gr, blk>>>(dst.u, src.u, src.v, src.w, vel.u, vel.v, vel.w, s.g_.solid,
                                 dt_step, g);
    advect_v_kernel<<<gr, blk>>>(dst.v, src.u, src.v, src.w, vel.u, vel.v, vel.w, s.g_.solid,
                                 dt_step, g);
    advect_w_kernel<<<gr, blk>>>(dst.w, src.u, src.v, src.w, vel.u, vel.v, vel.w, s.g_.solid,
                                 dt_step, g);
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
        double vu, vvel, vw;
        d_sample_velocity(uu, vv, ww, s[0], s[1], s[2], G.nx, G.ny, G.nz, G.dx, G.dy, G.dz, G.Lx,
                          G.Ly, G.Lz, vu, vvel, vw);
        d[0] = vu;
        d[1] = vvel;
        d[2] = vw;
        double g[9];
        d_velocity_gradient_at(s[0], s[1], s[2], uu, vv, ww, solid, G, g);
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
