/**
 * @file preconditioner_3d.h
 * @brief Abstract 3D preconditioner interface.
 * @author liutao
 * @date 2026-05-24
 */
#pragma once
#include "core/grid_3d.h"
#include <vector>
#include <string>

/**
 * @brief Abstract preconditioner interface for 3D solvers.
 *
 * Applies \f$M^{-1}\f$ to a 3D residual: \f$z = M^{-1} r\f$.
 * Depends on Grid3D for hierarchy construction (dimensions, solid mask).
 */
class Preconditioner3D {
public:
    virtual ~Preconditioner3D() = default;

    /**
     * @brief Apply the preconditioner.
     * @param g 3D grid (provides dimensions, spacing, solid mask).
     * @param r Input residual vector.
     * @param z Output: \f$z = M^{-1} r\f$.
     */
    virtual void apply(const Grid3D& g, const std::vector<double>& r,
                       std::vector<double>& z) = 0;

    /** @brief Human-readable preconditioner name. */
    virtual std::string name() const = 0;
};
