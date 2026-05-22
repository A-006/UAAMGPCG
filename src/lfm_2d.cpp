// ===========================================================================
// lfm_2d.cpp — Full 2D LFM fluid simulation (paper framework + Jacobi pressure solve)
//
// Compile: g++ -std=c++17 -O2 lfm_2d.cpp -o lfm_2d
// Run:     ./lfm_2d [karman|smoke] [NX] [nFrames]
//   Example: ./lfm_2d karman 128 200   → Karman vortex street, 128² grid
//
// [Paper component → Code mapping]
//   MAC staggered grid        → MACGrid2D class
//   Semi-Lagrangian advection (RK2) → advect_velocity()
//   Pressure projection       → pressure_projection()
//   Jacobi iteration          → solve_poisson_jacobi()  (replaces paper AMGPCG)
//   DoF marker                → is_solid boolean channel
//   Boundary conditions       → apply_boundary_conditions()
//   Leapfrog time step        → lfm_time_step()
//   VTK output                → write_vtk()
// ===========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>

// ===========================================================================
// Simulation parameters
// ===========================================================================
struct Config {
    std::string scenario = "karman";
    int    NX = 128, NY = 32;   // Number of grid cells
    double Lx = 4.0, Ly = 1.0;  // Domain [0,Lx]×[0,Ly]
    double U_inf = 1.0;         // Inflow velocity
    double cyl_cx = 1.0;        // Cylinder center x
    double cyl_cy = 0.5;        // Cylinder center y
    double cyl_R  = 0.1;        // Cylinder radius
    double Re = 200.0;          // Reynolds number
    double dt = 0.005;          // Time step size
    double t_end = 10.0;        // Simulation end time
    int    frame_skip = 10;     // Output one frame every N steps
    int    jacobi_iters = 2000; // Jacobi iteration count
    std::string out_dir = "output";
};

// ===========================================================================
// MAC staggered grid (2D)
// ===========================================================================
//
// NX×NY cells. Physical coordinates:
//   Cell (i,j): [(i-1)dx, i·dx] × [(j-1)dy, j·dy],  i∈[1,NX], j∈[1,NY]
//   u(i,j): x-direction velocity, on right face, location (i·dx, (j-0.5)dy), i∈[0,NX], j∈[1,NY]
//   v(i,j): y-direction velocity, on top face, location ((i-0.5)dx, j·dy), i∈[1,NX], j∈[0,NY]
//   p(i,j): pressure/scalar, at cell center,                    i∈[1,NX],   j∈[1,NY]
//
// All arrays include one layer of ghost cells (0 and N+1).
// Storage: 1D flattened, row-major (i innermost → j innermost).
//
class Grid {
public:
    int nx, ny;
    double dx, dy;
    std::vector<double> u, v, p;
    std::vector<bool>   solid;   // Solid marker (DoF = !solid)

    Grid(int nx_, int ny_, double lx, double ly)
        : nx(nx_), ny(ny_), dx(lx/nx_), dy(ly/ny_)
    {
        // u: (nx+1) faces × (ny+2) with ghost layer
        // v: (nx+2) with ghost layer × (ny+1) faces
        // p/solid: (nx+2) × (ny+2) with ghost layer
        u.resize((nx+1) * (ny+2), 0.0);
        v.resize((nx+2) * (ny+1), 0.0);
        p.resize((nx+2) * (ny+2), 0.0);
        solid.resize((nx+2) * (ny+2), false);
    }

    // ── Indexing and access ──
    // u: i∈[0,nx], j∈[0,ny+1]
    int  iu(int i, int j) const { return i*(ny+2) + j; }
    double  u_at(int i, int j) const { return u[iu(i,j)]; }
    double& u_at(int i, int j)       { return u[iu(i,j)]; }

    // v: i∈[0,nx+1], j∈[0,ny]
    int  iv(int i, int j) const { return i*(ny+1) + j; }
    double  v_at(int i, int j) const { return v[iv(i,j)]; }
    double& v_at(int i, int j)       { return v[iv(i,j)]; }

    // p/solid: i∈[0,nx+1], j∈[0,ny+1]
    int  ip(int i, int j) const { return i*(ny+2) + j; }
    double  p_at(int i, int j) const { return p[ip(i,j)]; }
    double& p_at(int i, int j)       { return p[ip(i,j)]; }
    bool  is_solid(int i, int j) const { return solid[ip(i,j)]; }
    void  set_solid(int i, int j)       { solid[ip(i,j)] = true; }
};

// ===========================================================================
// Utility functions
// ===========================================================================
inline double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Divergence ∇·u of cell (i,j)
inline double divergence(const Grid& g, int i, int j) {
    return (g.u_at(i,j) - g.u_at(i-1,j)) / g.dx
         + (g.v_at(i,j) - g.v_at(i,j-1)) / g.dy;
}

// ===========================================================================
// Bilinear interpolation (u field)
// ===========================================================================
// u(i,j) location: x = i*dx, y = (j-0.5)*dy
// Used to reconstruct u value at arbitrary (x,y)
double sample_u(const Grid& g, double x, double y) {
    x = clamp(x, 0.0, g.nx * g.dx);
    y = clamp(y, 0.0, g.ny * g.dy);

    // Continuous index
    double fi = x / g.dx;                        // i coordinate (u face at integer i)
    double fj = y / g.dy + 0.5;                  // j coordinate (u face at j-0.5)

    int i0 = (int)fi;
    int j0 = (int)fj;
    i0 = clamp(i0, 0, g.nx);
    j0 = clamp(j0, 0, g.ny+1);

    int i1 = std::min(i0+1, g.nx);
    int j1 = std::min(j0+1, g.ny+1);

    double wi = fi - i0;
    double wj = fj - j0;
    if (i1 == i0) wi = 0.0;
    if (j1 == j0) wj = 0.0;

    double v00 = g.u_at(i0,j0), v10 = g.u_at(i1,j0);
    double v01 = g.u_at(i0,j1), v11 = g.u_at(i1,j1);

    return (1-wi)*(1-wj)*v00 + wi*(1-wj)*v10 + (1-wi)*wj*v01 + wi*wj*v11;
}

// ===========================================================================
// Bilinear interpolation (v field)
// ===========================================================================
// v(i,j) location: x = (i-0.5)*dx, y = j*dy
double sample_v(const Grid& g, double x, double y) {
    x = clamp(x, 0.0, g.nx * g.dx);
    y = clamp(y, 0.0, g.ny * g.dy);

    double fi = x / g.dx + 0.5;                  // i coordinate (v face at i-0.5)
    double fj = y / g.dy;                        // j coordinate (v face at integer j)

    int i0 = (int)fi;
    int j0 = (int)fj;
    i0 = clamp(i0, 0, g.nx+1);
    j0 = clamp(j0, 0, g.ny);

    int i1 = std::min(i0+1, g.nx+1);
    int j1 = std::min(j0+1, g.ny);

    double wi = fi - i0;
    double wj = fj - j0;
    if (i1 == i0) wi = 0.0;
    if (j1 == j0) wj = 0.0;

    double v00 = g.v_at(i0,j0), v10 = g.v_at(i1,j0);
    double v01 = g.v_at(i0,j1), v11 = g.v_at(i1,j1);

    return (1-wi)*(1-wj)*v00 + wi*(1-wj)*v10 + (1-wi)*wj*v01 + wi*wj*v11;
}

// ===========================================================================
// RK2 backtrace (trace backward from (x,y) along velocity field by Δt)
// ===========================================================================
void backtrace(const Grid& g, double x, double y, double dt,
               double& xp, double& yp) {
    double u0 = sample_u(g, x, y);
    double v0 = sample_v(g, x, y);
    double xm = x - 0.5 * dt * u0;
    double ym = y - 0.5 * dt * v0;
    double um = sample_u(g, xm, ym);
    double vm = sample_v(g, xm, ym);
    xp = x - dt * um;
    yp = y - dt * vm;
}

// ===========================================================================
// Semi-Lagrangian advection (trace using old velocity field g_old, write result to g_new)
// ===========================================================================
void advect_velocity(const Grid& g_old, Grid& g_new, double dt) {
    int nx = g_old.nx, ny = g_old.ny;

    // Advect u (interior vertical faces i=1..nx-1)
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (g_old.is_solid(i,j) || g_old.is_solid(i+1,j)) {
                g_new.u_at(i,j) = 0.0;
                continue;
            }
            double xf = i * g_old.dx;
            double yf = (j - 0.5) * g_old.dy;
            double xp, yp;
            backtrace(g_old, xf, yf, dt, xp, yp);
            g_new.u_at(i,j) = sample_u(g_old, xp, yp);
        }
    }
    // Advect v (interior horizontal faces j=1..ny-1)
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j < ny; j++) {
            if (g_old.is_solid(i,j) || g_old.is_solid(i,j+1)) {
                g_new.v_at(i,j) = 0.0;
                continue;
            }
            double xf = (i - 0.5) * g_old.dx;
            double yf = j * g_old.dy;
            double xp, yp;
            backtrace(g_old, xf, yf, dt, xp, yp);
            g_new.v_at(i,j) = sample_v(g_old, xp, yp);
        }
    }
}

// ===========================================================================
// Jacobi solve Poisson equation ∇²p = rhs
// ===========================================================================
// 5-point Laplacian (non-uniform dx,dy):
//   (p_{i+1,j}+p_{i-1,j}-2p_{i,j})/dx² + (p_{i,j+1}+p_{i,j-1}-2p_{i,j})/dy² = rhs
//
// Jacobi scheme:
//   p_new = ( (pL+pR)/dx² + (pB+pT)/dy² - rhs ) / (2/dx²+2/dy²)
//
// Boundary conditions: outside domain ∂p/∂n=0 (Neumann), solid ∂p/∂n=0
void jacobi_solve(Grid& g, const std::vector<double>& rhs, int max_iter) {
    const double idx2 = 1.0 / (g.dx * g.dx);
    const double idy2 = 1.0 / (g.dy * g.dy);
    const double diag = 2.0 * (idx2 + idy2);
    const int    nx = g.nx, ny = g.ny;

    std::vector<double> pn(g.p.size());

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 1; i <= nx; i++) {
            for (int j = 1; j <= ny; j++) {
                if (g.is_solid(i,j)) {
                    pn[g.ip(i,j)] = 0.0;
                    continue;
                }

                // Neumann BC: solid neighbor or outside domain → use own p instead (∂p/∂n=0)
                double pL = (i > 1 && !g.is_solid(i-1,j)) ? g.p_at(i-1,j) : g.p_at(i,j);
                double pR = (i < nx && !g.is_solid(i+1,j)) ? g.p_at(i+1,j) : g.p_at(i,j);
                double pB = (j > 1 && !g.is_solid(i,j-1)) ? g.p_at(i,j-1) : g.p_at(i,j);
                double pT = (j < ny && !g.is_solid(i,j+1)) ? g.p_at(i,j+1) : g.p_at(i,j);

                double lap = (pL + pR) * idx2 + (pB + pT) * idy2;

                // Effective diagonal coefficient (accounting for Neumann BC reduced connections)
                double eff_d = diag;
                if (i == 1  || g.is_solid(i-1,j)) eff_d -= idx2;
                if (i == nx || g.is_solid(i+1,j)) eff_d -= idx2;
                if (j == 1  || g.is_solid(i,j-1)) eff_d -= idy2;
                if (j == ny || g.is_solid(i,j+1)) eff_d -= idy2;

                if (eff_d < 1e-15) {
                    pn[g.ip(i,j)] = 0.0;
                } else {
                    pn[g.ip(i,j)] = (lap - rhs[g.ip(i,j)]) / eff_d;
                }
            }
        }
        std::copy(pn.begin(), pn.end(), g.p.begin());
    }
}

// ===========================================================================
// Pressure projection: project velocity ũ onto divergence-free subspace
// ===========================================================================
//   1. Construct rhs = ∇·ũ / Δt
//   2. Solve ∇²p = rhs (Jacobi)
//   3. u = ũ - Δt·∇p
void pressure_projection(Grid& g, double dt, int jacobi_iters) {
    int nx = g.nx, ny = g.ny;

    // 1. RHS
    std::vector<double> rhs(g.p.size(), 0.0);
    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
            if (!g.is_solid(i,j))
                rhs[g.ip(i,j)] = divergence(g, i, j) / dt;

    // 2. Poisson solve
    std::fill(g.p.begin(), g.p.end(), 0.0);
    jacobi_solve(g, rhs, jacobi_iters);

    // 3. Velocity correction u ← ũ - Δt·∇p
    for (int i = 1; i < nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i+1,j)) continue;
            g.u_at(i,j) -= dt * (g.p_at(i+1,j) - g.p_at(i,j)) / g.dx;
        }
    }
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j < ny; j++) {
            if (g.is_solid(i,j) || g.is_solid(i,j+1)) continue;
            g.v_at(i,j) -= dt * (g.p_at(i,j+1) - g.p_at(i,j)) / g.dy;
        }
    }
}

// ===========================================================================
// Boundary conditions
// ===========================================================================
void apply_bc_karman(Grid& g, double U_inf) {
    int nx = g.nx, ny = g.ny;

    // Left boundary (inflow): u = U∞, v = 0
    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j) = U_inf;
        g.v_at(0, j) = 0.0;
    }
    // Right boundary (outflow): ∂u/∂x = 0, ∂v/∂x = 0
    for (int j = 1; j <= ny; j++) {
        g.u_at(nx, j) = g.u_at(nx-1, j);
        g.v_at(nx+1, j) = g.v_at(nx, j);
    }
    // Top/bottom walls: ∂u/∂y = 0 (slip), v = 0 (impermeable)
    for (int i = 0; i <= nx; i++) {
        g.u_at(i, 0)  = g.u_at(i, 1);
        g.u_at(i, ny+1) = g.u_at(i, ny);
    }
    for (int i = 1; i <= nx; i++) {
        g.v_at(i, 0)  = 0.0;
        g.v_at(i, ny) = 0.0;
    }
    // v ghost layer at inlet/outlet
    g.v_at(0, 0) = g.v_at(0, ny) = 0.0;
    g.v_at(nx+1, 0) = g.v_at(nx+1, ny) = 0.0;
}

void apply_bc_smoke(Grid& g) {
    int nx = g.nx, ny = g.ny;
    // Four walls no-slip
    for (int j = 1; j <= ny; j++) {
        g.u_at(0, j) = 0.0;
        g.u_at(nx, j) = 0.0;
    }
    for (int i = 1; i <= nx; i++) {
        g.v_at(i, 0) = 0.0;
        g.v_at(i, ny) = 0.0;
    }
    // Ghost layer
    for (int i = 0; i <= nx; i++) {
        g.u_at(i, 0) = -g.u_at(i, 1);          // No-slip: u_wall + u_ghost = 0
        g.u_at(i, ny+1) = -g.u_at(i, ny);
    }
    for (int j = 0; j <= ny; j++) {
        g.v_at(0, j) = -g.v_at(1, j);
        g.v_at(nx+1, j) = -g.v_at(nx, j);
    }
}

// Solid boundary (cylinder etc.): no-slip u=v=0
void apply_solid_bc(Grid& g) {
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            if (!g.is_solid(i,j)) continue;
            // Four faces
            if (i > 1 && !g.is_solid(i-1,j))       g.u_at(i-1, j) = 0.0;
            if (i < g.nx && !g.is_solid(i+1,j))    g.u_at(i, j)   = 0.0;
            if (j > 1 && !g.is_solid(i, j-1))      g.v_at(i, j-1) = 0.0;
            if (j < g.ny && !g.is_solid(i, j+1))   g.v_at(i, j)   = 0.0;
        }
    }
}

// ===========================================================================
// Obstacle setup (cylinder for Karman vortex street)
// ===========================================================================
void setup_cylinder(Grid& g, double cx, double cy, double R) {
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            double xc = (i - 0.5) * g.dx;
            double yc = (j - 0.5) * g.dy;
            if ((xc-cx)*(xc-cx) + (yc-cy)*(yc-cy) < R*R)
                g.set_solid(i, j);
        }
    }
}

// ===========================================================================
// LFM time step
// ===========================================================================
// Mid-step:
//   1. External forces (gravity/buoyancy)
//   2. Leapfrog advection (RK2 semi-Lagrangian)
//   3. Pressure projection (only once, this is the core advantage of LFM)
//
void lfm_time_step(Grid& g, double dt, const Config& cfg,
                   const Grid* g_bc = nullptr) {
    // ── 1. External forces ──
    if (cfg.scenario == "smoke") {
        double buoyancy = 5.0;
        for (int i = 1; i <= g.nx; i++)
            for (int j = 1; j <= g.ny; j++)
                if (!g.is_solid(i,j))
                    g.v_at(i,j) += dt * buoyancy;
    }

    // ── 2. Advection: backtrace using old velocity field, write result to new field ──
    const Grid& g_old = (g_bc ? *g_bc : g);  // Backtrace velocity field
    Grid g_adv = g_old;  // Copy backtrace field (including boundary conditions)
    // Reset interior velocities to 0, prepare to write advected fluxes
    // In fact, backtrace uses g_old's velocities, writes to g_adv
    // But g_adv is initialized as g_old, need to zero interior

    // Zero interior u,v of g_adv
    for (int i = 1; i < g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            g_adv.u_at(i,j) = 0.0;
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j < g.ny; j++)
            g_adv.v_at(i,j) = 0.0;

    advect_velocity(g_old, g_adv, dt);

    // Copy advection results back to g
    for (int i = 1; i < g.nx; i++)
        for (int j = 1; j <= g.ny; j++)
            g.u_at(i,j) = g_adv.u_at(i,j);
    for (int i = 1; i <= g.nx; i++)
        for (int j = 1; j < g.ny; j++)
            g.v_at(i,j) = g_adv.v_at(i,j);

    // Apply boundary conditions (after advection)
    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);

    // ── 3. Pressure projection ──
    pressure_projection(g, dt, cfg.jacobi_iters);

    // Re-apply boundary conditions after projection
    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);
}

// ===========================================================================
// VTK output (Legacy Structured Points)
// ===========================================================================
void write_vtk(const Grid& g, int frame, const Config& cfg) {
    std::ostringstream ss;
    ss << cfg.out_dir << "/frame_" << std::setw(5) << std::setfill('0') << frame << ".vtk";
    std::ofstream f(ss.str());
    if (!f) { std::cerr << "Cannot write " << ss.str() << "\n"; return; }

    f << std::scientific << std::setprecision(6);

    f << "# vtk DataFile Version 2.0\n";
    f << "LFM 2D Fluid - Frame " << frame << "\n";
    f << "ASCII\n";
    f << "DATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << (g.nx+1) << " " << (g.ny+1) << " 1\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << g.dx << " " << g.dy << " 1\n";

    int np = (g.nx+1) * (g.ny+1);
    f << "POINT_DATA " << np << "\n";

    // ── Velocity (interpolated at nodes) ──
    f << "VECTORS velocity float\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            // Node velocity = average of adjacent face velocities
            // i: left face u(i-1,j), right face u(i,j) → need to check range
            double u_sum = 0.0, v_sum = 0.0;
            int nu = 0, nv = 0;

            // u component: faces on left and right sides of node (i,j)
            if (i > 0 && j > 0 && j <= g.ny) {
                u_sum += g.u_at(i-1, j);
                nu++;
            }
            if (i < g.nx && j > 0 && j <= g.ny) {
                u_sum += g.u_at(i, j);
                nu++;
            }
            // v component: faces on top and bottom sides of node (i,j)
            if (j > 0 && i > 0 && i <= g.nx) {
                v_sum += g.v_at(i, j-1);
                nv++;
            }
            if (j < g.ny && i > 0 && i <= g.nx) {
                v_sum += g.v_at(i, j);
                nv++;
            }

            double uv = (nu > 0) ? u_sum / nu : 0.0;
            double vv = (nv > 0) ? v_sum / nv : 0.0;
            f << uv << " " << vv << " 0\n";
        }
    }

    // ── Vorticity (scalar, node interpolation) ──
    f << "SCALARS vorticity float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            double dvdx = 0.0, dudy = 0.0;
            if (i > 0 && i < g.nx && j > 0 && j <= g.ny)
                dvdx = (g.v_at(i+1,j) - g.v_at(i,j)) / g.dx;
            if (i > 0 && i <= g.nx && j > 0 && j < g.ny)
                dudy = (g.u_at(i,j+1) - g.u_at(i,j)) / g.dy;
            f << (dvdx - dudy) << "\n";
        }
    }

    // ── Divergence (scalar, cell-centered, take nearest cell value at nodes) ──
    f << "SCALARS divergence float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            int ci = clamp(i, 1, g.nx);
            int cj = clamp(j, 1, g.ny);
            f << divergence(g, ci, cj) << "\n";
        }
    }

    // ── Solid marker ──
    f << "SCALARS solid float 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int j = 0; j <= g.ny; j++) {
        for (int i = 0; i <= g.nx; i++) {
            int ci = clamp(i, 1, g.nx);
            int cj = clamp(j, 1, g.ny);
            f << (g.is_solid(ci,cj) ? 1.0 : 0.0) << "\n";
        }
    }

    f.close();
}

// ===========================================================================
// Status output
// ===========================================================================
void print_status(int step, double t, const Grid& g) {
    double max_u = 0.0, max_div = 0.0;
    for (int i = 1; i <= g.nx; i++) {
        for (int j = 1; j <= g.ny; j++) {
            if (g.is_solid(i,j)) continue;
            double uc = 0.5 * (g.u_at(i-1,j) + g.u_at(i,j));
            double vc = 0.5 * (g.v_at(i,j-1) + g.v_at(i,j));
            max_u = std::max(max_u, std::sqrt(uc*uc + vc*vc));
            max_div = std::max(max_div, std::abs(divergence(g,i,j)));
        }
    }
    std::cout << "  step=" << std::setw(6) << step
              << "  t=" << std::fixed << std::setprecision(4) << t
              << "  max|u|=" << std::setprecision(3) << max_u
              << "  max|div|=" << std::scientific << std::setprecision(3) << max_div
              << std::fixed << "\n";
}

// ===========================================================================
// Main program
// ===========================================================================
#ifndef TEST_MODE
int main(int argc, char* argv[]) {
    Config cfg;

    if (argc > 1) cfg.scenario = argv[1];
    if (argc > 2) cfg.NX = std::atoi(argv[2]);
    if (argc > 3) cfg.t_end = std::atof(argv[3]);

    if (cfg.scenario == "karman") {
        cfg.Lx = 4.0; cfg.Ly = 1.0;
        cfg.NY = cfg.NX / 4;
        if (cfg.NY < 16) cfg.NY = 16;
        cfg.out_dir = "output_karman";
        cfg.dt = 0.5 * (cfg.Lx / cfg.NX) / cfg.U_inf;  // CFL
        cfg.jacobi_iters = 2000;
    } else if (cfg.scenario == "smoke") {
        cfg.Lx = 1.0; cfg.Ly = 1.0;
        cfg.NY = cfg.NX;
        cfg.out_dir = "output_smoke";
        cfg.dt = 0.005;
        cfg.jacobi_iters = 2000;
    } else {
        std::cerr << "Usage: lfm_2d [karman|smoke] [NX] [t_end]\n";
        return 1;
    }

    mkdir(cfg.out_dir.c_str(), 0755);

    std::cout << "+" << std::string(52, '-') << "+\n";
    std::cout << "| LFM 2D Fluid Simulation — " << cfg.scenario
              << std::string(26 - (int)cfg.scenario.size(), ' ') << "|\n";
    std::cout << "| grid: " << cfg.NX << "×" << cfg.NY
              << "  cells=" << cfg.NX * cfg.NY
              << std::string(16, ' ') << "|\n";
    std::cout << "| dx=" << std::setprecision(4) << (cfg.Lx/cfg.NX)
              << "  dy=" << (cfg.Ly/cfg.NY)
              << "  dt=" << cfg.dt
              << "  t_end=" << cfg.t_end << "        |\n";
    std::cout << "| Jacobi iters=" << cfg.jacobi_iters
              << "  output: " << cfg.out_dir << "/frame_*.vtk |\n";
    std::cout << "+" << std::string(52, '-') << "+\n\n";

    // ── Initialize grid ──
    Grid g(cfg.NX, cfg.NY, cfg.Lx, cfg.Ly);

    // Initial conditions
    if (cfg.scenario == "karman") {
        setup_cylinder(g, cfg.cyl_cx, cfg.cyl_cy, cfg.cyl_R);
        // Uniform inflow throughout domain
        for (int i = 0; i <= cfg.NX; i++)
            for (int j = 1; j <= cfg.NY; j++)
                g.u_at(i,j) = cfg.U_inf;
    }
    // smoke: initially stationary

    if (cfg.scenario == "karman") apply_bc_karman(g, cfg.U_inf);
    else                          apply_bc_smoke(g);
    apply_solid_bc(g);

    // ── Time stepping ──
    int nsteps = (int)(cfg.t_end / cfg.dt);
    auto t0 = std::chrono::high_resolution_clock::now();
    double t = 0.0;

    // Backup initial velocity field for LFM advection backtrace
    Grid g_prev = g;

    for (int step = 0; step < nsteps; step++) {
        // LFM mid-step: backtrace using g_prev velocities, write result to g
        lfm_time_step(g, cfg.dt, cfg, &g_prev);

        // Save current velocity to g_prev (for next step's backtrace)
        g_prev = g;

        t += cfg.dt;

        if (step % cfg.frame_skip == 0) {
            print_status(step, t, g);
            write_vtk(g, step / cfg.frame_skip, cfg);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\nDone: " << nsteps << " steps in " << elapsed << " s";
    std::cout << " (" << (elapsed/nsteps*1000) << " ms/step)\n";
    std::cout << "Open " << cfg.out_dir << "/frame_*.vtk in ParaView\n";

    return 0;
}
#endif
