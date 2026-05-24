#pragma once
#include "core/grid.h"
#include <vector>
#include <string>

/// Cylinder force computation on MAC grid.
/// Computes pressure + viscous forces on solid cylinder boundary.
struct CylinderForce {
    double Fx = 0, Fy = 0;  // total force components
    double Fp_x = 0, Fp_y = 0;  // pressure force
    double Fv_x = 0, Fv_y = 0;  // viscous force

    /// Compute force coefficients Cd, Cl.
    /// D = cylinder diameter, U_inf = freestream velocity, rho = 1.
    double Cd(double U_inf, double D) const { return 2.0 * Fx / (U_inf * U_inf * D); }
    double Cl(double U_inf, double D) const { return 2.0 * Fy / (U_inf * U_inf * D); }
};

/// Compute force on all solid cells in the grid.
/// Identifies solid cells that have at least one fluid neighbor,
/// and integrates pressure + viscous stress on fluid-solid faces.
CylinderForce computeForce(const Grid& g, double dt, double U_inf, double Re,
                            double cyl_cx, double cyl_cy, double cyl_R);

/// Write Cd/Cl time history to CSV file.
void writeForceHistory(const std::string& filename,
                       const std::vector<double>& time,
                       const std::vector<double>& Cd,
                       const std::vector<double>& Cl);

/// Estimate Strouhal number from Cl signal using zero-crossing.
/// St = f * D / U_inf
double estimateStrouhal(const std::vector<double>& time,
                         const std::vector<double>& Cl,
                         double U_inf, double D);
