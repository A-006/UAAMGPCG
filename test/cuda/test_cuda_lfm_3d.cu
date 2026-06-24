// ════════════════════════════════════════════════════════════════════
// GPU 3D LFM kernels — validation against the CPU LFMSimulator3D golden
// reference. Each phase appends a test; all use the codebase's GPU-vs-CPU
// cross-check pattern (run CPU ref + GPU, assert max|cpu-gpu| < tol).
//
// P0: free-slip box BC + pressure projection vs CPU FreeSlipAllFaces3D /
//     PressureProjection3D.
// ════════════════════════════════════════════════════════════════════
#include "solver/cuda/cuda_lfm_3d.h"
#include "core/grid_3d.h"
#include "config/config.h"
#include "numerics/bc/patches_3d.h"
#include "numerics/pressure/pressure_3d.h"
#include "simulator/lfm/lfm_simulator_3d.h"
#include "simulator/lfm/cuda_lfm_simulator_3d.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include "solver/cuda_pcg_solver_3d.h"
#include "solver/factory_3d.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* msg, double val = 0) {
    printf("  [%s] %s", ok ? "PASS" : "FAIL", msg);
    if (val != 0)
        printf("  (%.3e)", val);
    printf("\n");
    if (!ok)
        g_fail++;
}

static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); i++)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// Smooth, clearly-divergent analytic field sampled at MAC face positions.
static void fill_field(int nx, int ny, int nz, double dx, double dy, double dz,
                       std::vector<double>& hu, std::vector<double>& hv, std::vector<double>& hw) {
    hu.assign(lfm_u_size(nx, ny, nz), 0.0);
    hv.assign(lfm_v_size(nx, ny, nz), 0.0);
    hw.assign(lfm_w_size(nx, ny, nz), 0.0);
    auto fu = [](double x, double y, double z) { return std::sin(2 * x) + 0.3 * y; };
    auto fv = [](double x, double y, double z) { return std::cos(2 * y) + 0.2 * z; };
    auto fw = [](double x, double y, double z) { return 0.1 * x + 0.15 * z; };
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 0; i <= nx; i++)
                hu[lfm_iu(i, j, k, nx, ny)] = fu(i * dx, (j - 0.5) * dy, (k - 0.5) * dz);
    for (int k = 1; k <= nz; k++)
        for (int j = 0; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                hv[lfm_iv(i, j, k, nx, ny)] = fv((i - 0.5) * dx, j * dy, (k - 0.5) * dz);
    for (int k = 0; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                hw[lfm_iw(i, j, k, nx, ny)] = fw((i - 0.5) * dx, (j - 0.5) * dy, k * dz);
}

static double host_max_div(const Grid3D& g) {
    double m = 0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++)
                if (!g.is_solid(i, j, k))
                    m = std::max(m, std::abs(g.divergence(i, j, k)));
    return m;
}

static void test_p0() {
    const int nx = 24, ny = 24, nz = 24;
    const double L = 1.0, dx = L / nx, dy = L / ny, dz = L / nz, dt = 0.01;
    const int iters = 400;
    const double tol = 1e-10;

    printf("=== P0: free-slip BC + projection (GPU vs CPU) ===\n");

    std::vector<double> hu, hv, hw;
    fill_field(nx, ny, nz, dx, dy, dz, hu, hv, hw);

    // ── GPU state ──
    CudaLFMState3D s;
    s.fp32_march = false; // bit-exact GPU-vs-CPU reference: force the FP64 marcher
    s.allocate(nx, ny, nz, dx, dy, dz, /*n_steps=*/2);
    std::vector<char> solid(lfm_p_size(nx, ny, nz), 0);
    s.upload_solid(solid);
    s.upload_velocity(hu, hv, hw);

    // ── (1) BC: GPU vs CPU free_slip_box ──
    Grid3D gcpu(nx, ny, nz, L, L, L);
    gcpu.u = hu;
    gcpu.v = hv;
    gcpu.w = hw;
    bc::free_slip_box().apply(gcpu);

    lfm_apply_free_slip_box(s, s.cur);
    cudaDeviceSynchronize();
    std::vector<double> gu(hu.size()), gv(hv.size()), gw(hw.size());
    s.download_velocity(gu, gv, gw);
    double bc_du = max_abs_diff(gcpu.u, gu);
    double bc_dv = max_abs_diff(gcpu.v, gv);
    double bc_dw = max_abs_diff(gcpu.w, gw);
    check(std::max({bc_du, bc_dv, bc_dw}) < 1e-12, "free-slip BC matches CPU bit-for-bit",
          std::max({bc_du, bc_dv, bc_dw}));

    // ── (2) Projection: GPU vs CPU PressureProjection3D ──
    auto cpu_solver = Factory3D::create("pcg_uaamg");
    PressureProjection3D::project(gcpu, dt, *cpu_solver, iters, tol);

    lfm_project(s, s.cur, dt, iters, tol);
    cudaDeviceSynchronize();
    s.download_velocity(gu, gv, gw);

    double pu = max_abs_diff(gcpu.u, gu), pv = max_abs_diff(gcpu.v, gv),
           pw = max_abs_diff(gcpu.w, gw);
    double pmax = std::max({pu, pv, pw});
    check(pmax < 1e-5, "projected velocity matches CPU (within solver tol)", pmax);

    // GPU result should itself be ~divergence-free.
    Grid3D ggpu(nx, ny, nz, L, L, L);
    ggpu.u = gu;
    ggpu.v = gv;
    ggpu.w = gw;
    double gdiv = host_max_div(ggpu);
    double cdiv = host_max_div(gcpu);
    printf("    CPU max|div|=%.3e  GPU max|div|=%.3e\n", cdiv, gdiv);
    check(gdiv < 1e-4, "GPU projection is divergence-free", gdiv);

    s.free();
}

// ── P1: semi-Lagrangian advection vs CPU LFMSimulator3D::rk2_advect ──
static void test_p1() {
    const int nx = 24, ny = 24, nz = 24;
    const double L = 1.0, dx = L / nx, dy = L / ny, dz = L / nz, dt_step = 0.02;
    printf("\n=== P1: rk2 semi-Lagrangian advection (GPU vs CPU) ===\n");

    std::vector<double> su, sv, sw;
    fill_field(nx, ny, nz, dx, dy, dz, su, sv, sw);
    // velocity field (advecting field): a swirl, distinct from src
    std::vector<double> vu(su.size()), vv(sv.size()), vw(sw.size());
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 0; i <= nx; i++)
                vu[lfm_iu(i, j, k, nx, ny)] = -((j - 0.5) * dy - 0.5);
    for (int k = 1; k <= nz; k++)
        for (int j = 0; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                vv[lfm_iv(i, j, k, nx, ny)] = ((i - 0.5) * dx - 0.5);
    // vw stays 0

    // CPU reference
    Config cfg;
    cfg.dim = 3;
    cfg.NX = nx;
    cfg.NY = ny;
    cfg.NZ = nz;
    cfg.Lx = cfg.Ly = cfg.Lz = L;
    cfg.Re                   = 0;
    cfg.time_integrator      = "lfm";
    LFMSimulator3D sim(cfg, Factory3D::create("cg"));
    Grid3D srcG(nx, ny, nz, L, L, L);
    srcG.u = su;
    srcG.v = sv;
    srcG.w = sw;
    Grid3D dstG = srcG; // boundary faces retained from src, like the GPU init
    sim.rk2_advect_public(dstG, srcG, vu, vv, vw, dt_step);

    // GPU
    CudaLFMState3D s;
    s.fp32_march = false; // bit-exact GPU-vs-CPU reference: force the FP64 marcher
    s.allocate(nx, ny, nz, dx, dy, dz, 2);
    std::vector<char> solid(lfm_p_size(nx, ny, nz), 0);
    s.upload_solid(solid);
    s.upload_to(s.A, su, sv, sw); // src
    s.upload_to(s.B, vu, vv, vw); // vel
    s.upload_to(s.C, su, sv, sw); // dst init = src (match CPU dst=src copy)
    lfm_rk2_advect(s, s.C, s.A, s.B, dt_step);
    cudaDeviceSynchronize();
    std::vector<double> gu(su.size()), gv(sv.size()), gw(sw.size());
    s.download_from(s.C, gu, gv, gw);

    double du = max_abs_diff(dstG.u, gu), dv = max_abs_diff(dstG.v, gv),
           dw = max_abs_diff(dstG.w, gw);
    double m = std::max({du, dv, dw});
    check(m < 1e-11, "advected velocity matches CPU rk2_advect", m);
    s.free();
}

// ── P2: flow-map RK4 marching vs CPU rk4_march_forward/backward ──
static void test_p2() {
    const int nx = 24, ny = 24, nz = 24;
    const double L = 1.0, dx = L / nx, dy = L / ny, dz = L / nz, dt = 0.02;
    printf("\n=== P2: flow-map RK4 marching (GPU vs CPU) ===\n");

    // Solid-body rotation about z: du/dy=-1, dv/dx=+1 → F rotates (exercises ∇u·F).
    std::vector<double> vu(lfm_u_size(nx, ny, nz), 0.0), vv(lfm_v_size(nx, ny, nz), 0.0),
        vw(lfm_w_size(nx, ny, nz), 0.0);
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 0; i <= nx; i++)
                vu[lfm_iu(i, j, k, nx, ny)] = -((j - 0.5) * dy - 0.5);
    for (int k = 1; k <= nz; k++)
        for (int j = 0; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                vv[lfm_iv(i, j, k, nx, ny)] = ((i - 0.5) * dx - 0.5);

    Config cfg;
    cfg.dim = 3;
    cfg.NX = nx;
    cfg.NY = ny;
    cfg.NZ = nz;
    cfg.Lx = cfg.Ly = cfg.Lz = L;
    cfg.Re                   = 0;
    cfg.time_integrator      = "lfm";
    LFMSimulator3D sim(cfg, Factory3D::create("cg"));
    FlowMap3D& fm = sim.flow_map();

    CudaLFMState3D s;
    s.fp32_march = false; // bit-exact GPU-vs-CPU reference: force the FP64 marcher
    s.allocate(nx, ny, nz, dx, dy, dz, 2);
    std::vector<char> solid(lfm_p_size(nx, ny, nz), 0);
    s.upload_solid(solid);
    s.upload_to(s.A, vu, vv, vw);

    auto cmp = [&](const double* dptr, const std::vector<double>& cpu) {
        std::vector<double> h(cpu.size());
        cudaMemcpy(h.data(), dptr, cpu.size() * sizeof(double), cudaMemcpyDeviceToHost);
        return max_abs_diff(h, cpu);
    };

    // ── Forward: 3 steps ──
    fm.set_identity();
    lfm_set_identity(s);
    for (int n = 0; n < 3; n++) {
        sim.rk4_march_forward_public(vu, vv, vw, dt);
        lfm_rk4_march_forward(s, s.A, dt);
    }
    cudaDeviceSynchronize();
    double ef = std::max({cmp(s.phi_x, fm.phi_x), cmp(s.phi_y, fm.phi_y), cmp(s.phi_z, fm.phi_z)});
    double eF = std::max({cmp(s.F[0], fm.F00), cmp(s.F[1], fm.F01), cmp(s.F[2], fm.F02),
                          cmp(s.F[3], fm.F10), cmp(s.F[4], fm.F11), cmp(s.F[5], fm.F12),
                          cmp(s.F[6], fm.F20), cmp(s.F[7], fm.F21), cmp(s.F[8], fm.F22)});
    check(ef < 1e-10, "forward march Φ matches CPU", ef);
    check(eF < 1e-10, "forward march F (Jacobian) matches CPU", eF);

    // ── Backward: 3 steps ──
    fm.set_backward_identity();
    lfm_set_backward_identity(s);
    for (int n = 0; n < 3; n++) {
        sim.rk4_march_backward_public(vu, vv, vw, -dt);
        lfm_rk4_march_backward(s, s.A, -dt);
    }
    cudaDeviceSynchronize();
    double eb = std::max({cmp(s.psi_x, fm.psi_x), cmp(s.psi_y, fm.psi_y), cmp(s.psi_z, fm.psi_z)});
    double eT = std::max({cmp(s.T[0], fm.T00), cmp(s.T[1], fm.T01), cmp(s.T[2], fm.T02),
                          cmp(s.T[3], fm.T10), cmp(s.T[4], fm.T11), cmp(s.T[5], fm.T12),
                          cmp(s.T[6], fm.T20), cmp(s.T[7], fm.T21), cmp(s.T[8], fm.T22)});
    check(eb < 1e-10, "backward march Ψ matches CPU", eb);
    check(eT < 1e-10, "backward march T (Jacobian) matches CPU", eT);

    // ── FIX① per-axis (staggered-face) march: forward (φ,F) + backward (ψ,T) ──
    // Forward face map: 3 steps with +dt; backward: 3 steps with -dt.
    sim.face_set_forward_identity_public();
    lfm_face_set_forward_identity(s);
    for (int n = 0; n < 3; n++) {
        sim.face_march_forward_public(vu, vv, vw, dt);
        lfm_face_march_forward(s, s.A, dt);
    }
    sim.face_set_backward_identity_public();
    lfm_face_set_backward_identity(s);
    for (int n = 0; n < 3; n++) {
        sim.face_march_backward_public(vu, vv, vw, -dt);
        lfm_face_march_backward(s, s.A, -dt);
    }
    cudaDeviceSynchronize();
    const auto& cfu = sim.face_map_u();
    const auto& cfv = sim.face_map_v();
    const auto& cfw = sim.face_map_w();
    // Forward map position + covector row across all three axes.
    double efp = std::max({cmp(s.fmu.fx, cfu.fx), cmp(s.fmu.fy, cfu.fy), cmp(s.fmu.fz, cfu.fz),
                           cmp(s.fmv.fx, cfv.fx), cmp(s.fmw.fz, cfw.fz)});
    double efF = std::max({cmp(s.fmu.f0, cfu.f0), cmp(s.fmu.f1, cfu.f1), cmp(s.fmu.f2, cfu.f2),
                           cmp(s.fmv.f0, cfv.f0), cmp(s.fmv.f1, cfv.f1), cmp(s.fmw.f2, cfw.f2)});
    double ebp = std::max({cmp(s.fmu.bx, cfu.bx), cmp(s.fmv.by, cfv.by), cmp(s.fmw.bz, cfw.bz)});
    double ebT = std::max({cmp(s.fmu.t0, cfu.t0), cmp(s.fmu.t1, cfu.t1), cmp(s.fmu.t2, cfu.t2),
                           cmp(s.fmv.t1, cfv.t1), cmp(s.fmw.t2, cfw.t2)});
    check(efp < 1e-10, "FIX① forward face map φ matches CPU", efp);
    check(efF < 1e-10, "FIX① forward face covector F matches CPU", efF);
    check(ebp < 1e-10, "FIX① backward face map ψ matches CPU", ebp);
    check(ebT < 1e-10, "FIX① backward face covector T matches CPU", ebT);
    s.free();
}

// ── P3: viscous, pullback, forward-pullback vs CPU ──
static void test_p3() {
    const int nx = 24, ny = 24, nz = 24;
    const double L = 1.0, dx = L / nx, dy = L / ny, dz = L / nz, dt = 0.02;
    const long fs = (long)nx * ny * nz;
    printf("\n=== P3: impulse chain (viscous / pullback / forward-pullback) ===\n");

    Config cfg;
    cfg.dim = 3;
    cfg.NX = nx;
    cfg.NY = ny;
    cfg.NZ = nz;
    cfg.Lx = cfg.Ly = cfg.Lz = L;
    cfg.Re                   = 200;
    cfg.U_inf                = 1.0;
    cfg.cyl_R                = 0.1;
    cfg.time_integrator      = "lfm";
    const double mu = cfg.U_inf * 2 * cfg.cyl_R / cfg.Re;
    LFMSimulator3D sim(cfg, Factory3D::create("cg"));
    FlowMap3D& fm = sim.flow_map();

    CudaLFMState3D s;
    s.fp32_march = false; // bit-exact GPU-vs-CPU reference: force the FP64 marcher
    s.allocate(nx, ny, nz, dx, dy, dz, 2);
    std::vector<char> solid(lfm_p_size(nx, ny, nz), 0);
    s.upload_solid(solid);

    // Common non-identity flow map (marched; validated in P2).
    std::vector<double> sw_u(lfm_u_size(nx, ny, nz), 0.0), sw_v(lfm_v_size(nx, ny, nz), 0.0),
        sw_w(lfm_w_size(nx, ny, nz), 0.0);
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 0; i <= nx; i++)
                sw_u[lfm_iu(i, j, k, nx, ny)] = -((j - 0.5) * dy - 0.5);
    for (int k = 1; k <= nz; k++)
        for (int j = 0; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                sw_v[lfm_iv(i, j, k, nx, ny)] = ((i - 0.5) * dx - 0.5);
    s.upload_to(s.A, sw_u, sw_v, sw_w);
    fm.set_identity();
    lfm_set_identity(s);
    for (int n = 0; n < 2; n++) {
        sim.rk4_march_forward_public(sw_u, sw_v, sw_w, dt);
        lfm_rk4_march_forward(s, s.A, dt);
    }
    fm.set_backward_identity();
    lfm_set_backward_identity(s);
    for (int n = 0; n < 2; n++) {
        sim.rk4_march_backward_public(sw_u, sw_v, sw_w, -dt);
        lfm_rk4_march_backward(s, s.A, -dt);
    }
    // FIX① per-face flow maps (same swirl), for the face pullback test below.
    sim.face_set_forward_identity_public();
    lfm_face_set_forward_identity(s);
    for (int n = 0; n < 2; n++) {
        sim.face_march_forward_public(sw_u, sw_v, sw_w, dt);
        lfm_face_march_forward(s, s.A, dt);
    }
    sim.face_set_backward_identity_public();
    lfm_face_set_backward_identity(s);
    for (int n = 0; n < 2; n++) {
        sim.face_march_backward_public(sw_u, sw_v, sw_w, -dt);
        lfm_face_march_backward(s, s.A, -dt);
    }

    auto cmp = [&](const double* dptr, const std::vector<double>& cpu) {
        std::vector<double> h(cpu.size());
        cudaMemcpy(h.data(), dptr, cpu.size() * sizeof(double), cudaMemcpyDeviceToHost);
        return max_abs_diff(h, cpu);
    };

    // ── (1) viscous ──
    std::vector<double> fu, fv, fw;
    fill_field(nx, ny, nz, dx, dy, dz, fu, fv, fw);
    Grid3D gf(nx, ny, nz, L, L, L);
    gf.u = fu;
    gf.v = fv;
    gf.w = fw;
    std::vector<double> cvu, cvv, cvw;
    sim.compute_viscous_public(gf, cvu, cvv, cvw);
    s.upload_to(s.B, fu, fv, fw);
    lfm_compute_viscous(s, s.B, mu);
    cudaDeviceSynchronize();
    double ev = std::max({cmp(s.visc_x, cvu), cmp(s.visc_y, cvv), cmp(s.visc_z, cvw)});
    check(ev < 1e-13, "viscous force matches CPU", ev);

    (void)fs;
    const int us = lfm_u_size(nx, ny, nz), vs = lfm_v_size(nx, ny, nz), ws = lfm_w_size(nx, ny, nz);

    // ── (2) FIX① per-face pullback m_a = T_a·u0(ψ_a), staggered, backward map ──
    std::vector<double> cm_u(us, 0.0), cm_v(vs, 0.0), cm_w(ws, 0.0);
    sim.face_pullback_public(fu, fv, fw, cm_u, cm_v, cm_w, /*fwd=*/false);
    s.upload_to(s.u0, fu, fv, fw);
    lfm_face_pullback(s, s.u0, s.mface, /*fwd=*/false);
    cudaDeviceSynchronize();
    double ep = std::max(
        {cmp(s.mface.u, cm_u), cmp(s.mface.v, cm_v), cmp(s.mface.w, cm_w)});
    check(ep < 1e-13, "FIX① face pullback (backward) matches CPU", ep);

    // ── (3) FIX① forward face pullback û_a = F_a·m(φ_a), staggered ──
    std::vector<double> mfu(us, 0.0), mfv(vs, 0.0), mfw(ws, 0.0);
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 0; i <= nx; i++)
                mfu[lfm_iu(i, j, k, nx, ny)] = std::sin(0.3 * i) + 0.1 * k;
    for (int k = 1; k <= nz; k++)
        for (int j = 0; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                mfv[lfm_iv(i, j, k, nx, ny)] = std::cos(0.2 * j) + 0.05 * i;
    for (int k = 0; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++)
                mfw[lfm_iw(i, j, k, nx, ny)] = 0.05 * (i + j) + 0.02 * k;
    std::vector<double> cuu(us, 0.0), cuv(vs, 0.0), cuw(ws, 0.0);
    sim.face_pullback_public(mfu, mfv, mfw, cuu, cuv, cuw, /*fwd=*/true);
    s.upload_to(s.A, mfu, mfv, mfw); // reuse A as a staggered source field
    lfm_face_pullback(s, s.A, s.mhat, /*fwd=*/true);
    cudaDeviceSynchronize();
    double efp = std::max(
        {cmp(s.mhat.u, cuu), cmp(s.mhat.v, cuv), cmp(s.mhat.w, cuw)});
    check(efp < 1e-13, "FIX① forward face pullback matches CPU", efp);

    s.free();
}

// ── P4: full LFM cycle (GPU CudaLFMSimulator3D vs CPU LFMSimulator3D) ──
static double field_ke(const Grid3D& g) {
    double ke = 0;
    for (int k = 1; k <= g.nz; k++)
        for (int j = 1; j <= g.ny; j++)
            for (int i = 1; i <= g.nx; i++) {
                double uc = 0.5 * (g.u_at(i, j, k) + g.u_at(i - 1, j, k));
                double vc = 0.5 * (g.v_at(i, j, k) + g.v_at(i, j - 1, k));
                double wc = 0.5 * (g.w_at(i, j, k) + g.w_at(i, j, k - 1));
                ke += 0.5 * (uc * uc + vc * vc + wc * wc);
            }
    return ke * g.dx * g.dy * g.dz;
}

static void setup_ring(Grid3D& g) {
    scenarios::VortexRing vr;
    vr.center      = {0.5, 0.5, 0.4};
    vr.axis        = {0, 0, 1};
    vr.radius      = 0.18;
    vr.core        = 0.05;
    vr.circulation = 1.0;
    vr.n_segments  = 200;
    scenarios::add_vortex_ring(g, vr);
}

// Full-GPU CudaLFMSimulator3D vs the already-validated CPU LFMSimulator3D driven
// by the SAME GPU Poisson solver (CudaPCGSolver3D). Using the identical solver on
// both sides isolates the GPU kernels/orchestration from solver-convergence
// differences — so the only remaining variation is GPU vs CPU per-cell arithmetic
// (bit-identical with FMA disabled, per P0–P3).
static void test_p4(double Re, const char* tag, bool clamp = false) {
    const int nx = 32, ny = 32, nz = 32;
    const double L = 1.0;
    printf("\n=== P4 (%s, Re=%.0f%s): full GPU LFM cycle vs CPU sim + same GPU solver ===\n", tag, Re,
           clamp ? ", BFECC clamp" : "");

    Config cfg;
    cfg.dim = 3;
    cfg.NX = nx;
    cfg.NY = ny;
    cfg.NZ = nz;
    cfg.Lx = cfg.Ly = cfg.Lz = L;
    cfg.Re                   = Re; // Re>0 exercises the viscous accumulate / path-integral chain
    cfg.U_inf                = 1.0;
    cfg.cyl_R                = 0.1;
    cfg.dt                   = 0.25 * (L / nx);
    cfg.solve_iters          = 300;
    cfg.solve_tol            = 1e-7;
    cfg.time_integrator      = "lfm";
    cfg.lfm_cycle_steps      = 3; // ≥3 also exercises the leapfrog main loop + path integral
    cfg.lfm_bfecc_clamp      = clamp;
    cfg.lfm_march_fp32       = false; // bit-exact GPU-vs-CPU reference: force the FP64 marcher

    // CPU orchestration, GPU Poisson solve (same CudaPCG3D the GPU sim uses).
    LFMSimulator3D cpu(cfg, std::make_unique<CudaPCGSolver3D>(true));
    setup_ring(cpu.mutable_grid());
    cpu.set_boundary_manager(bc::free_slip_box());

    // GPU device-resident sim, same IC.
    CudaLFMSimulator3D gpu(cfg);
    setup_ring(gpu.mutable_grid());
    gpu.commit();

    double ke0 = field_ke(cpu.grid());
    cpu.step();
    gpu.step();

    // Post-cycle FIX① per-face backward flow map ψ_u and face impulse m should
    // match the CPU reference bit-for-bit.
    auto cmpfm = [&](const double* dptr, const std::vector<double>& cpuv) {
        std::vector<double> h(cpuv.size());
        cudaMemcpy(h.data(), dptr, cpuv.size() * sizeof(double), cudaMemcpyDeviceToHost);
        return max_abs_diff(h, cpuv);
    };
    const auto& fmu = cpu.face_map_u();
    double dpsi = std::max({cmpfm(gpu.state().fmu.bx, fmu.bx), cmpfm(gpu.state().fmu.by, fmu.by),
                            cmpfm(gpu.state().fmu.bz, fmu.bz)});
    double dm = std::max({cmpfm(gpu.state().mface.u, cpu.impulse_x()),
                          cmpfm(gpu.state().mface.v, cpu.impulse_y()),
                          cmpfm(gpu.state().mface.w, cpu.impulse_z())});
    check(dpsi < 1e-12, "post-cycle per-face backward map ψ_u matches CPU", dpsi);
    check(dm < 1e-12, "post-cycle face impulse m matches CPU", dm);

    const Grid3D& gc = cpu.grid();
    const Grid3D& gg = gpu.grid();
    double dmax = std::max({max_abs_diff(gc.u, gg.u), max_abs_diff(gc.v, gg.v),
                            max_abs_diff(gc.w, gg.w)});
    double keC = field_ke(gc), keG = field_ke(gg);
    double divC = host_max_div(gc), divG = host_max_div(gg);
    printf("    CPU max|div|=%.3e  GPU max|div|=%.3e\n", divC, divG);
    printf("    KE0=%.5f  CPU KE=%.5f (%.2f%%)  GPU KE=%.5f (%.2f%%)\n", ke0, keC,
           100 * keC / ke0, keG, 100 * keG / ke0);
    check(dmax < 1e-9, "full cycle GPU velocity matches CPU (same GPU solver)", dmax);
    check(std::abs(keC - keG) / ke0 < 1e-9, "kinetic energy matches CPU");
}

int main() {
    // This suite validates the GPU solver bit-for-bit against the FP64 CPU
    // reference (tol 1e-9), so it must exercise the FP64 projection path. The
    // production default is the faster FP32 tile-native solve (validated
    // end-to-end via |div|max, not against the FP64 CPU golden).
    setenv("PCG_FP64", "1", 1);
    test_p0();
    test_p1();
    test_p2();
    test_p3();
    test_p4(0.0, "inviscid");
    test_p4(300.0, "viscous");        // validates the GPU viscous accumulate / path-integral chain
    test_p4(0.0, "inviscid", true);   // validates the GPU BFECC clamp matches CPU bit-for-bit
    printf(g_fail == 0 ? "\nALL PASS\n" : "\n%d FAILED\n", g_fail);
    return g_fail ? 1 : 0;
}
