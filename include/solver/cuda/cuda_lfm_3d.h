#pragma once
#include "solver/cuda/cuda_common_3d.h"
#include "solver/cuda/cuda_pcg_3d.h"
#include <vector>

// ════════════════════════════════════════════════════════════════════
// Device-resident 3D LFM state + kernel launch API.
//
// This is the GPU port of src/simulator/lfm_simulator_3d.cpp. All fields live
// on the device for the whole reinitialization cycle; the host only launches
// kernels and the device-resident Poisson solve, and copies back to a host
// Grid3D once per cycle for output. The CPU LFMSimulator3D remains the golden
// reference that every kernel here is validated against (test_cuda_lfm_3d.cu).
//
// Index helpers below MUST match include/core/mesh_3d.h (Mesh3D::iu/iv/iw/ip)
// and include/simulator/flow_map_3d.h (FlowMap3D::idx) bit-for-bit, otherwise
// the CPU↔GPU cross-checks fail.
// ════════════════════════════════════════════════════════════════════

// ── MAC face / cell / flow-map index helpers (mirror Mesh3D + FlowMap3D) ──
__host__ __device__ inline int lfm_iu(int i, int j, int k, int nx, int ny) {
    return i + j * (nx + 1) + k * (nx + 1) * (ny + 2);
}
__host__ __device__ inline int lfm_iv(int i, int j, int k, int nx, int ny) {
    return i + j * (nx + 2) + k * (nx + 2) * (ny + 1);
}
__host__ __device__ inline int lfm_iw(int i, int j, int k, int nx, int ny) {
    return i + j * (nx + 2) + k * (nx + 2) * (ny + 2);
}
__host__ __device__ inline int lfm_ip(int i, int j, int k, int nx, int ny) {
    return i + j * (nx + 2) + k * (nx + 2) * (ny + 2);
}
// Flow-map (interior cells only): (i-1)+(j-1)*nx+(k-1)*nx*ny, i,j,k ∈ [1,nx]×[1,ny]×[1,nz]
__host__ __device__ inline long lfm_fm(int i, int j, int k, int nx, int ny) {
    return (long)(i - 1) + (long)(j - 1) * nx + (long)(k - 1) * (long)nx * ny;
}

inline int lfm_u_size(int nx, int ny, int nz) { return (nx + 1) * (ny + 2) * (nz + 2); }
inline int lfm_v_size(int nx, int ny, int nz) { return (nx + 2) * (ny + 1) * (nz + 2); }
inline int lfm_w_size(int nx, int ny, int nz) { return (nx + 2) * (ny + 2) * (nz + 1); }
inline int lfm_p_size(int nx, int ny, int nz) { return (nx + 2) * (ny + 2) * (nz + 2); }
inline long lfm_fm_size(int nx, int ny, int nz) { return (long)nx * ny * nz; }

// One MAC velocity field (three face components) living on the device.
struct CudaVel3D {
    double *u = nullptr, *v = nullptr, *w = nullptr;
};

// All device buffers for one LFM simulation. Allocated once; only u/v/w/p are
// copied back to host per cycle (for VTK). n_steps = cfg.lfm_cycle_steps.
struct CudaLFMState3D {
    int nx = 0, ny = 0, nz = 0;
    double dx = 0, dy = 0, dz = 0;
    double Lx = 0, Ly = 0, Lz = 0;
    int n_steps = 0;

    // Face flow-map marching / sampling precision. true (default) = FP32 math
    // (author-faithful, ~order faster on consumer GPUs whose FP64 throughput is
    // 1/64). The bit-exact GPU-vs-CPU tests set this false for the FP64 path.
    bool fp32_march = true;
    // EXPERIMENT: when true (and fp32_march is also true), the flow-map sampling
    // scratch is stored as __half and the 27-point B-spline interpolation runs in
    // packed half2. Gated by cfg.lfm_sample_fp16 / env LFM_FP16. Off → FP32 path.
    bool sample_fp16 = false;

    CudaGrid3D g_{};               // solid mask + grid metadata for the Poisson solve
    CudaPCG3D pcg_{};              // device UAAMG-PCG (null-space safe)
    double *d_p = nullptr;          // pressure (p_size)
    double *d_rhs = nullptr;        // Poisson rhs (p_size)

    CudaVel3D cur, u0, A, B, C;     // current grid, u0, and three work velocity fields
    std::vector<CudaVel3D> vb;      // vel_buffer[n_steps]

    // Flow map (interior-sized): forward Φ + Jacobian F, backward Ψ + T, midpoints.
    double *phi_x = nullptr, *phi_y = nullptr, *phi_z = nullptr;
    double *F[9] = {};
    double *psi_x = nullptr, *psi_y = nullptr, *psi_z = nullptr;
    double *T[9] = {};
    double *phi_mid_x = nullptr, *phi_mid_y = nullptr, *phi_mid_z = nullptr;
    double *F_mid[9] = {};

    double *m_x = nullptr, *m_y = nullptr, *m_z = nullptr;       // impulse (interior)
    double *visc_x = nullptr, *visc_y = nullptr, *visc_z = nullptr; // cell-centered viscous
    double *e_x = nullptr, *e_y = nullptr, *e_z = nullptr;       // error-correction (interior)
    double *uhat_x = nullptr, *uhat_y = nullptr, *uhat_z = nullptr;

    // ── FIX①: per-axis (staggered-face) flow maps + face impulse ──
    // Per face axis a∈{u,v,w}: backward pos ψ_a (3) + covector row T_a (3), and
    // forward pos φ_a (3) + covector row F_a (3), all on that axis' MAC face grid.
    // Mirrors the CPU FaceFlowMap; bp/fp = positions, T/F = covector rows.
    struct CudaFaceFlowMap {
        double *bx = nullptr, *by = nullptr, *bz = nullptr; // backward ψ
        double *t0 = nullptr, *t1 = nullptr, *t2 = nullptr; // covector T
        double *fx = nullptr, *fy = nullptr, *fz = nullptr; // forward φ
        double *f0 = nullptr, *f1 = nullptr, *f2 = nullptr; // covector F
    };
    CudaFaceFlowMap fmu{}, fmv{}, fmw{};
    CudaVel3D mface{}; // impulse on faces (u_size/v_size/w_size)
    CudaVel3D mhat{};  // forward-pullback scratch û0 (face-sized)
    CudaVel3D merr{};  // error / correction scratch (face-sized)

    // FP32 scratch copies of a MAC velocity field, used only by the fp32_march
    // production sampling path. The flow-map sampling kernels gather double
    // velocity values and convert each to float per load — on consumer GPUs that
    // FP64→FP32 convert runs on the 1/64-rate FP64 pipe and was the #1 hotspot of
    // face_march/face_pullback (ncu: FP64 pipe ~24%). Pre-converting the whole
    // field once to these float buffers turns the hot gather into a pure FP32
    // load, removing the per-sample FP64 convert. The FP64 (bit-exact test) path
    // keeps reading the double arrays directly. Sized like one CudaVel3D.
    float *vu32 = nullptr, *vv32 = nullptr, *vw32 = nullptr; // velocity (march)
    float *su32 = nullptr, *sv32 = nullptr, *sw32 = nullptr; // source  (pullback/advect-src)

    // EXPERIMENT (sample_fp16): __half scratch copies of the same MAC fields. The
    // sampling gather then reads FP16 (half the gather bytes vs FP32) and the
    // 27-point B-spline runs in packed half2. Declared as void* here so the header
    // need not include <cuda_fp16.h>; cast to __half* inside the .cu.
    void *vu16 = nullptr, *vv16 = nullptr, *vw16 = nullptr; // velocity (march)
    void *su16 = nullptr, *sv16 = nullptr, *sw16 = nullptr; // source  (pullback/advect-src)

    void allocate(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_, int n_steps_);
    void free();

    // Host helpers: copy the current velocity field to/from host MAC arrays.
    void upload_velocity(const std::vector<double>& hu, const std::vector<double>& hv,
                         const std::vector<double>& hw);
    void download_velocity(std::vector<double>& hu, std::vector<double>& hv,
                           std::vector<double>& hw) const;
    void upload_solid(const std::vector<char>& hsolid); // p_size, 1=solid

    // Generic copy to/from an arbitrary device velocity field (cur/u0/A/B/C/vb).
    void upload_to(CudaVel3D q, const std::vector<double>& hu, const std::vector<double>& hv,
                   const std::vector<double>& hw) const;
    void download_from(CudaVel3D q, std::vector<double>& hu, std::vector<double>& hv,
                       std::vector<double>& hw) const;
};

// ── Launch API (host-callable; implemented across the cuda_lfm_3d_*.cu files) ──
// P0: projection + boundary conditions.
void lfm_apply_free_slip_box(CudaLFMState3D& s, CudaVel3D vel);
// Freestream box BC: prescribe (Ux,Uy,Uz) on all six walls + no-slip solid.
void lfm_apply_freestream_box(CudaLFMState3D& s, CudaVel3D vel, double Ux, double Uy, double Uz);
void lfm_project(CudaLFMState3D& s, CudaVel3D vel, double dt, int iters, double tol);
// P1: semi-Lagrangian advection (dst ← advect src by vel for dt_step).
void lfm_rk2_advect(CudaLFMState3D& s, CudaVel3D dst, CudaVel3D src, CudaVel3D vel, double dt_step);
// P2: flow-map RK4 marching + identity init.
void lfm_set_identity(CudaLFMState3D& s);
void lfm_set_backward_identity(CudaLFMState3D& s);
void lfm_rk4_march_forward(CudaLFMState3D& s, CudaVel3D vel, double dt_march);
void lfm_rk4_march_backward(CudaLFMState3D& s, CudaVel3D vel, double dt_march);
// P3: impulse chain.
void lfm_compute_viscous(CudaLFMState3D& s, CudaVel3D vel, double mu); // → s.visc_*
void lfm_accumulate_to_u0(CudaLFMState3D& s, CudaVel3D u0, const double* cx, const double* cy,
                          const double* cz, double coeff);
void lfm_accumulate_path_integral(CudaLFMState3D& s, CudaVel3D u0, double coeff); // uses s.visc_*
void lfm_save_flow_map_state(CudaLFMState3D& s);
void lfm_compute_midpoints(CudaLFMState3D& s);
void lfm_pullback_impulse(CudaLFMState3D& s, CudaVel3D u0); // → s.m_*
void lfm_forward_pullback(CudaLFMState3D& s, const double* mx, const double* my, const double* mz,
                          double* ox, double* oy, double* oz);
void lfm_error_correction(CudaLFMState3D& s, CudaVel3D u0,
                          bool clamp = false); // updates s.m_*; clamp=BFECC neighbour clamp
void lfm_gauge_writeback(CudaLFMState3D& s, CudaVel3D vel); // vel ← face-avg(s.m_*)
// Device-to-device velocity copy (dst ← src).
void lfm_copy_vel(CudaLFMState3D& s, CudaVel3D dst, CudaVel3D src);

// ── FIX① per-axis (staggered-face) flow-map launch API ──
void lfm_face_set_forward_identity(CudaLFMState3D& s);
void lfm_face_set_backward_identity(CudaLFMState3D& s);
void lfm_face_march_forward(CudaLFMState3D& s, CudaVel3D vel, double dt_march);
void lfm_face_march_backward(CudaLFMState3D& s, CudaVel3D vel, double dt_march);
// Per-face pullback dst_a = T_a·src(ψ_a) (fwd=false: backward map; fwd=true: forward map).
void lfm_face_pullback(CudaLFMState3D& s, CudaVel3D src, CudaVel3D dst, bool fwd);
// Face BFECC error correction; writes s.mface and then vel ← mface (no gauge avg).
void lfm_face_error_correction(CudaLFMState3D& s, CudaVel3D u0, CudaVel3D vel, bool clamp);
