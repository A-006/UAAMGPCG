/**
 * @file identity_preconditioner_3d.h
 * @brief Identity preconditioner \f$M = I\f$ — reduces PCG to plain CG for 3D.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "solver/preconditioner/preconditioner_3d.h"

/**
 * @brief Identity preconditioner: \f$z = r\f$ (no preconditioning).
 *
 * Makes CG a special case of PCG with \f$M = I\f$.
 */
class IdentityPreconditioner3D : public Preconditioner3D {
public:
    /** @brief Copies r to z unchanged. */
    void apply(const Grid3D&, const std::vector<double>& r,
               std::vector<double>& z) override { z = r; }
    std::string name() const override { return "none"; }
};
