#pragma once
#include "core/grid.h"
#include <vector>
#include <string>
#include <cmath>

/// Abstract cylinder representation. Two implementations:
///   StairStepCylinder — binary solid/fluid (original, backward-compatible)
///   SmoothCylinder    — continuous solid fraction φ∈[0,1] via smoothstep
class CylinderModel {
public:
    virtual ~CylinderModel() = default;

    /// Solid fraction at position (x,y): 0=fluid, 1=solid
    virtual double solidFraction(double x, double y) const = 0;

    /// Modify Laplacian coefficients and mark solid cells.
    /// Smooth: cells with φ>0.999 marked as solid; transition zone gets variable coeffs.
    virtual void applyToLaplacian(Grid& g,
                                  std::vector<double>& diag,
                                  std::vector<double>& off_x,
                                  std::vector<double>& off_y) = 0;

    /// Enforce no-slip boundary: modify velocity field near cylinder.
    virtual void applyToVelocity(Grid& g) const = 0;

    /// Return name for logging
    virtual std::string name() const = 0;
};

// ═══════════════════════════════════════════════════════════
// StairStep: original binary solid marking
// ═══════════════════════════════════════════════════════════
class StairStepCylinder : public CylinderModel {
public:
    StairStepCylinder(double cx, double cy, double R, double dx, double dy);
    double solidFraction(double x, double y) const override;
    void applyToLaplacian(Grid& g, std::vector<double>& diag,
                          std::vector<double>& off_x,
                          std::vector<double>& off_y) override;
    void applyToVelocity(Grid& g) const override;
    std::string name() const override { return "StairStep"; }

private:
    double cx_, cy_, R_, dx_, dy_;
};

// ═══════════════════════════════════════════════════════════
// SmoothCylinder: continuous solid fraction via smoothstep
// φ(x,y) = smoothstep( (r-R) / transition_width )
// transition_width = dx (one cell)
// ═══════════════════════════════════════════════════════════
class SmoothCylinder : public CylinderModel {
public:
    SmoothCylinder(double cx, double cy, double R, double dx, double dy,
                   double transition_width = 0.0);
    double solidFraction(double x, double y) const override;
    void applyToLaplacian(Grid& g, std::vector<double>& diag,
                          std::vector<double>& off_x,
                          std::vector<double>& off_y) override;
    void applyToVelocity(Grid& g) const override;
    std::string name() const override { return "Smooth(IB)"; }

private:
    double cx_, cy_, R_, dx_, dy_, tw_;  // tw = transition width
    static double smoothstep(double t);   // Hermite interpolation
};
