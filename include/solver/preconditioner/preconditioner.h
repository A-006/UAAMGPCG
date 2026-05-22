/**
 * @file preconditioner.h
 * @brief Abstract preconditioner interface.
 * @author liutao
 * @date 2026-05-22
 */
#pragma once
#include "core/grid.h"
#include <vector>
#include <string>

/**
 * @brief Abstract preconditioner interface.
 *
 * Applies \f$M^{-1}\f$ to a residual: \f$z = M^{-1} r\f$.
 * Depends on Grid for hierarchy construction (dimensions, solid mask).
 */
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    /**
     * @brief Apply the preconditioner.
     * @param g Grid (provides dimensions, spacing, solid mask).
     * @param r Input residual vector.
     * @param z Output: \f$z = M^{-1} r\f$.
     */
    virtual void apply(const Grid& g, const std::vector<double>& r,
                       std::vector<double>& z) = 0;

    /** @brief Human-readable preconditioner name. */
    virtual std::string name() const = 0;
};
