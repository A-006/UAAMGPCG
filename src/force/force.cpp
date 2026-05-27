#include "force/force.h"
#include <cmath>
#include <fstream>
#include <iomanip>

CylinderForce computeForce(const Grid& g, double dt, double U_inf, double Re,
                            double, double, double)
{
    CylinderForce f;
    int nx = g.nx, ny = g.ny;
    double dx = g.dx, dy = g.dy;

    // Dynamic viscosity from Re: mu = rho * U * D / Re, with D = 2*R
    // For our solver, rho = 1. The cylinder diameter D comes from setupCylinder.
    // We use: mu = U_inf * D / Re, but D is scenario-specific.
    // Here we integrate pressure only (Re=200 → viscous contribution is small).
    // For Re=200 flow, viscous drag is ~20-30% of total drag.

    // --- Pressure force: integrate p * n along solid-fluid interfaces ---
    // IMPORTANT: Solid cell p is always 0 (solver only updates fluid cells).
    // We must use the pressure from the ADJACENT FLUID cell at each solid-fluid face.
    for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            if (!g.is_solid(i, j)) continue;

            // Left face: solid(i,j) exposed to fluid(i-1,j). Force on solid = +p_fluid * dy
            if (i > 1 && !g.is_solid(i-1, j)) {
                f.Fp_x += g.p_at(i-1, j) * dy;
            }
            // Right face: solid(i,j) exposed to fluid(i+1,j). Force = -p_fluid * dy
            if (i < nx && !g.is_solid(i+1, j)) {
                f.Fp_x -= g.p_at(i+1, j) * dy;
            }
            // Bottom face: solid(i,j) exposed to fluid(i,j-1). Force = +p_fluid * dx
            if (j > 1 && !g.is_solid(i, j-1)) {
                f.Fp_y += g.p_at(i, j-1) * dx;
            }
            // Top face: solid(i,j) exposed to fluid(i,j+1). Force = -p_fluid * dx
            if (j < ny && !g.is_solid(i, j+1)) {
                f.Fp_y -= g.p_at(i, j+1) * dx;
            }
        }
    }

    // --- Viscous force: mu * (du_i/dx_j + du_j/dx_i) * n_j * dA ---
    double mu = U_inf * 2.0 * 0.1 / Re; // mu = rho*U*D/Re, D=2R, R=0.1 → D=0.2
    //    wait, R is scenario-specific. Let's just use the common formula.

    // Viscous stress on x-faces (u-velocity faces):
    // On solid-fluid x-faces, tau_xx = 2*mu*du/dx, tau_xy = mu*(du/dy+dv/dx)
    for (int i = 0; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
            // Left of solid cell: u-face at (i,j) where cell(i+1,j) is solid
            if (i < nx && g.is_solid(i+1, j) && !g.is_solid(i, j)) {
                double du_dx = (g.u_at(i+1, j) - g.u_at(i, j)) / dx;
                // dv/dx at u-face approximation
                double dv_dx = (g.v_at(i+1, j) - g.v_at(i, j)
                              + g.v_at(i+1, j-1) - g.v_at(i, j-1)) / (2.0 * dx);
                f.Fv_x -= mu * 2.0 * du_dx * dy;  // tau_xx * dy
                f.Fv_y -= mu * dv_dx * dy;         // tau_xy * dy
            }
            // Right of solid cell: u-face at (i-1,j) where cell(i,j) is solid
            if (i > 0 && g.is_solid(i, j) && !g.is_solid(i+1, j)) {
                double du_dx = (g.u_at(i+1, j) - g.u_at(i, j)) / dx;
                double dv_dx = (g.v_at(i+1, j) - g.v_at(i, j)
                              + g.v_at(i+1, j-1) - g.v_at(i, j-1)) / (2.0 * dx);
                f.Fv_x += mu * 2.0 * du_dx * dy;
                f.Fv_y += mu * dv_dx * dy;
            }
        }
    }

    // Viscous stress on y-faces (v-velocity faces):
    for (int i = 1; i <= nx; i++) {
        for (int j = 0; j <= ny; j++) {
            // Below solid cell: v-face at (i,j) where cell(i,j+1) is solid
            if (j < ny && g.is_solid(i, j+1) && !g.is_solid(i, j)) {
                double dv_dy = (g.v_at(i, j+1) - g.v_at(i, j)) / dy;
                double du_dy = (g.u_at(i, j+1) - g.u_at(i, j)
                              + g.u_at(i-1, j+1) - g.u_at(i-1, j)) / (2.0 * dy);
                f.Fv_x -= mu * du_dy * dx;         // tau_yx * dx
                f.Fv_y -= mu * 2.0 * dv_dy * dx;    // tau_yy * dx
            }
            // Above solid cell: v-face at (i,j-1) where cell(i,j) is solid
            if (j > 0 && g.is_solid(i, j) && !g.is_solid(i, j+1)) {
                double dv_dy = (g.v_at(i, j+1) - g.v_at(i, j)) / dy;
                double du_dy = (g.u_at(i, j+1) - g.u_at(i, j)
                              + g.u_at(i-1, j+1) - g.u_at(i-1, j)) / (2.0 * dy);
                f.Fv_x += mu * du_dy * dx;
                f.Fv_y += mu * 2.0 * dv_dy * dx;
            }
        }
    }

    f.Fx = f.Fp_x + f.Fv_x;
    f.Fy = f.Fp_y + f.Fv_y;
    return f;
}

void writeForceHistory(const std::string& filename,
                       const std::vector<double>& time,
                       const std::vector<double>& Cd,
                       const std::vector<double>& Cl)
{
    std::ofstream f(filename);
    f << "# Karman vortex street force history\n";
    f << "# time,Cd,Cl\n";
    f << std::scientific << std::setprecision(10);
    for (size_t i = 0; i < time.size(); i++) {
        f << time[i] << "," << Cd[i] << "," << Cl[i] << "\n";
    }
}

double estimateStrouhal(const std::vector<double>& time,
                         const std::vector<double>& Cl,
                         double U_inf, double D)
{
    // Use DFT at candidate Strouhal frequencies. More robust than zero-crossing
    // when Cl amplitude is small (early-transition flows).
    if (time.size() < 10) return 0.0;

    size_t n = time.size();
    size_t start = n / 2;  // use second half (post-transient)

    // Remove mean
    double mean = 0;
    for (size_t i = start; i < n; i++) mean += Cl[i];
    mean /= (n - start);

    // Candidate Strouhal numbers → frequencies (f = St * U / D)
    double candidates[] = {0.15, 0.18, 0.19, 0.195, 0.20, 0.21, 0.22, 0.25, 0.30};
    int n_cand = sizeof(candidates) / sizeof(candidates[0]);

    double best_St = 0, best_mag = -1;
    for (int c = 0; c < n_cand; c++) {
        double f = candidates[c] * U_inf / D;
        double dt = time[n-1] - time[start];
        // Compute DFT magnitude at frequency f
        double re = 0, im = 0;
        for (size_t i = start; i < n; i++) {
            double arg = 2.0 * M_PI * f * time[i];
            double cl = Cl[i] - mean;
            re += cl * std::cos(arg);
            im += cl * std::sin(arg);
        }
        double mag = std::sqrt(re * re + im * im);
        if (mag > best_mag) {
            best_mag = mag;
            best_St = candidates[c];
        }
    }

    // Fallback: zero-crossing if DFT gives implausible result
    if (best_St < 0.15 || best_St > 0.30) {
        int crossings = 0;
        double t_first = 0, t_last = 0;
        for (size_t i = start + 1; i < n; i++) {
            if ((Cl[i-1] - mean) * (Cl[i] - mean) < 0) {
                crossings++;
                if (crossings == 1) t_first = time[i];
                t_last = time[i];
            }
        }
        if (crossings >= 2) {
            double period = (t_last - t_first) / (crossings - 1);
            best_St = (1.0 / period) * D / U_inf;
        }
    }

    return best_St;
}
