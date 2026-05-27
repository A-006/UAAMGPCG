#include "simulator/simulator_3d.h"
#include "advection/advection_3d.h"
#include "bc/patches_3d.h"
#include "pressure/pressure_3d.h"

ChorinSimulator3D::ChorinSimulator3D(const Config& cfg, std::unique_ptr<Solver3D> solver)
    : cfg_(cfg),
      grid_(cfg.NX, cfg.NY, cfg.NZ, cfg.Lx, cfg.Ly, cfg.Lz),
      prev_(cfg.NX, cfg.NY, cfg.NZ, cfg.Lx, cfg.Ly, cfg.Lz),
      solver_(std::move(solver))
{
    apply_bc();
    prev_ = grid_;
}

void ChorinSimulator3D::apply_bc() {
    // Default: free-slip box + immersed solid. Scenarios needing periodic
    // BCs (e.g., decaying isotropic turbulence) can replace by constructing
    // a BoundaryManager3D externally and applying it instead.
    bc::FreeSlipAllFaces3D walls;          walls.apply(grid_);
    bc::NoSlipImmersedSolid3D solid;       solid.apply(grid_);
}

void ChorinSimulator3D::advect() {
    Grid3D g_adv = prev_;
    // Zero interior of advected target so backtrace results populate them
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    for (int k = 1; k <= nz; k++) for (int j = 1; j <= ny; j++) for (int i = 1; i < nx; i++) g_adv.u_at(i,j,k) = 0;
    for (int k = 1; k <= nz; k++) for (int j = 1; j < ny; j++) for (int i = 1; i <= nx; i++) g_adv.v_at(i,j,k) = 0;
    for (int k = 1; k < nz; k++) for (int j = 1; j <= ny; j++) for (int i = 1; i <= nx; i++) g_adv.w_at(i,j,k) = 0;

    AdvectionScheme3D::advect(prev_, g_adv, cfg_.dt);

    for (int k = 1; k <= nz; k++) for (int j = 1; j <= ny; j++) for (int i = 1; i < nx; i++)
        grid_.u_at(i,j,k) = g_adv.u_at(i,j,k);
    for (int k = 1; k <= nz; k++) for (int j = 1; j < ny; j++) for (int i = 1; i <= nx; i++)
        grid_.v_at(i,j,k) = g_adv.v_at(i,j,k);
    for (int k = 1; k < nz; k++) for (int j = 1; j <= ny; j++) for (int i = 1; i <= nx; i++)
        grid_.w_at(i,j,k) = g_adv.w_at(i,j,k);
}

void ChorinSimulator3D::diffuse() {
    if (cfg_.Re <= 0) return;
    double nu = cfg_.U_inf * 2.0 * cfg_.cyl_R / cfg_.Re;
    if (nu <= 0) return;
    int nx = grid_.nx, ny = grid_.ny, nz = grid_.nz;
    double idx2 = 1.0/(grid_.dx*grid_.dx);
    double idy2 = 1.0/(grid_.dy*grid_.dy);
    double idz2 = 1.0/(grid_.dz*grid_.dz);
    auto u = grid_.u, v = grid_.v, w = grid_.w;
    // u
    for (int k = 1; k <= nz; k++) for (int j = 1; j <= ny; j++) for (int i = 1; i < nx; i++) {
        if (grid_.is_solid(i,j,k) || grid_.is_solid(i+1,j,k)) continue;
        double uC = u[grid_.iu(i,j,k)];
        double uL = (i>1) ? u[grid_.iu(i-1,j,k)] : uC;
        double uR = (i<nx-1) ? u[grid_.iu(i+1,j,k)] : uC;
        double uB = (j>1) ? u[grid_.iu(i,j-1,k)] : uC;
        double uT = (j<ny) ? u[grid_.iu(i,j+1,k)] : uC;
        double uF = (k>1) ? u[grid_.iu(i,j,k-1)] : uC;
        double uK = (k<nz) ? u[grid_.iu(i,j,k+1)] : uC;
        grid_.u_at(i,j,k) += cfg_.dt * nu *
                            ((uL + uR - 2*uC) * idx2
                           + (uB + uT - 2*uC) * idy2
                           + (uF + uK - 2*uC) * idz2);
    }
    // v
    for (int k = 1; k <= nz; k++) for (int j = 1; j < ny; j++) for (int i = 1; i <= nx; i++) {
        if (grid_.is_solid(i,j,k) || grid_.is_solid(i,j+1,k)) continue;
        double vC = v[grid_.iv(i,j,k)];
        double vL = (i>1) ? v[grid_.iv(i-1,j,k)] : vC;
        double vR = (i<nx) ? v[grid_.iv(i+1,j,k)] : vC;
        double vB = (j>1) ? v[grid_.iv(i,j-1,k)] : vC;
        double vT = (j<ny-1) ? v[grid_.iv(i,j+1,k)] : vC;
        double vF = (k>1) ? v[grid_.iv(i,j,k-1)] : vC;
        double vK = (k<nz) ? v[grid_.iv(i,j,k+1)] : vC;
        grid_.v_at(i,j,k) += cfg_.dt * nu *
                            ((vL + vR - 2*vC) * idx2
                           + (vB + vT - 2*vC) * idy2
                           + (vF + vK - 2*vC) * idz2);
    }
    // w
    for (int k = 1; k < nz; k++) for (int j = 1; j <= ny; j++) for (int i = 1; i <= nx; i++) {
        if (grid_.is_solid(i,j,k) || grid_.is_solid(i,j,k+1)) continue;
        double wC = w[grid_.iw(i,j,k)];
        double wL = (i>1) ? w[grid_.iw(i-1,j,k)] : wC;
        double wR = (i<nx) ? w[grid_.iw(i+1,j,k)] : wC;
        double wB = (j>1) ? w[grid_.iw(i,j-1,k)] : wC;
        double wT = (j<ny) ? w[grid_.iw(i,j+1,k)] : wC;
        double wF = (k>1) ? w[grid_.iw(i,j,k-1)] : wC;
        double wK = (k<nz-1) ? w[grid_.iw(i,j,k+1)] : wC;
        grid_.w_at(i,j,k) += cfg_.dt * nu *
                            ((wL + wR - 2*wC) * idx2
                           + (wB + wT - 2*wC) * idy2
                           + (wF + wK - 2*wC) * idz2);
    }
}

void ChorinSimulator3D::project() {
    PressureProjection3D::project(grid_, cfg_.dt, *solver_, cfg_.solve_iters, cfg_.solve_tol);
}

void ChorinSimulator3D::step() {
    prev_ = grid_;
    advect();
    apply_bc();
    diffuse();
    apply_bc();
    project();
    apply_bc();
    t_ += cfg_.dt;
    step_++;
}
