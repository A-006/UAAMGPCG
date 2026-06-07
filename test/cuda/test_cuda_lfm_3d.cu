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
#include "simulator/lfm_simulator_3d.h"
#include "simulator/cuda_lfm_simulator_3d.h"
#include "simulator/scenarios/3d/vortex_ring.h"
#include "solver/cuda_pcg_solver_3d.h"
#include "solver/factory_3d.h"
#include <cmath>
#include <cstdio>
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

    // ── (2) pullback m = T^T u0(Ψ) ──
    Grid3D u0G(nx, ny, nz, L, L, L);
    u0G.u = fu;
    u0G.v = fv;
    u0G.w = fw;
    sim.pullback_impulse_public(u0G);
    s.upload_to(s.u0, fu, fv, fw);
    lfm_pullback_impulse(s, s.u0);
    cudaDeviceSynchronize();
    double ep = std::max({cmp(s.m_x, sim.impulse_x()), cmp(s.m_y, sim.impulse_y()),
                          cmp(s.m_z, sim.impulse_z())});
    check(ep < 1e-13, "pullback impulse matches CPU", ep);

    // ── (3) forward pullback û = F^T m(Φ) ──
    std::vector<double> mfx(fs), mfy(fs), mfz(fs);
    for (int k = 1; k <= nz; k++)
        for (int j = 1; j <= ny; j++)
            for (int i = 1; i <= nx; i++) {
                long id = lfm_fm(i, j, k, nx, ny);
                mfx[id] = std::sin(0.3 * i) + 0.1 * k;
                mfy[id] = std::cos(0.2 * j);
                mfz[id] = 0.05 * (i + j);
            }
    std::vector<double> cux(fs), cuy(fs), cuz(fs);
    sim.forward_pullback_public(mfx, mfy, mfz, cux, cuy, cuz);
    cudaMemcpy(s.m_x, mfx.data(), fs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.m_y, mfy.data(), fs * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(s.m_z, mfz.data(), fs * sizeof(double), cudaMemcpyHostToDevice);
    lfm_forward_pullback(s, s.m_x, s.m_y, s.m_z, s.uhat_x, s.uhat_y, s.uhat_z);
    cudaDeviceSynchronize();
    double efp = std::max({cmp(s.uhat_x, cux), cmp(s.uhat_y, cuy), cmp(s.uhat_z, cuz)});
    check(efp < 1e-13, "forward pullback matches CPU", efp);

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

    // Post-cycle impulse m and backward flow map Ψ should match the CPU reference.
    auto cmpfm = [&](const double* dptr, const std::vector<double>& cpuv) {
        std::vector<double> h(cpuv.size());
        cudaMemcpy(h.data(), dptr, cpuv.size() * sizeof(double), cudaMemcpyDeviceToHost);
        return max_abs_diff(h, cpuv);
    };
    FlowMap3D& fmc = cpu.flow_map();
    double dpsi = std::max({cmpfm(gpu.state().psi_x, fmc.psi_x), cmpfm(gpu.state().psi_y, fmc.psi_y),
                            cmpfm(gpu.state().psi_z, fmc.psi_z)});
    double dm = std::max({cmpfm(gpu.state().m_x, cpu.impulse_x()),
                          cmpfm(gpu.state().m_y, cpu.impulse_y()),
                          cmpfm(gpu.state().m_z, cpu.impulse_z())});
    check(dpsi < 1e-12, "post-cycle backward flow map Ψ matches CPU", dpsi);
    check(dm < 1e-12, "post-cycle impulse m matches CPU", dm);

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
