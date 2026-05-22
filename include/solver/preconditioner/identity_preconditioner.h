/**
 * @file identity_preconditioner.h
 * @brief Identity preconditioner \f$M = I\f$ — reduces PCG to plain CG.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "solver/preconditioner/preconditioner.h"

/**
 * @brief Identity preconditioner: \f$z = r\f$ (no preconditioning).
 *
 * Makes CG a special case of PCG with \f$M = I\f$.
 */
class IdentityPreconditioner : public Preconditioner {
public:
    /** @brief Copies r to z unchanged. */
    void apply(const Grid&, const std::vector<double>& r,
               std::vector<double>& z) override { z = r; }
    std::string name() const override { return "none"; }
};
