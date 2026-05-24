#pragma once
#include "solver/preconditioner/preconditioner.h"
#include <vector>

class UAAMGPreconditioner : public Preconditioner {
public:
    void apply(const Grid& g, const std::vector<double>& r,
               std::vector<double>& z) override;
    std::string name() const override { return "UAAMG"; }

private:
    struct Level {
        int nx, ny;
        double dx, dy;
        std::vector<double> p, b;
        std::vector<bool> solid;
    };
    std::vector<Level> levels_;
    int cached_nx_ = -1, cached_ny_ = -1;

    void buildHierarchy(const Grid& g);
    void vCycle(int level, int nlevels);
    static void restrictSolid(const Level& fine, Level& coarse);
    static void smooth(Level& L, int sweeps);
    static void restrictResidual(const Level& fine, Level& coarse);
    static void prolongateAdd(const Level& coarse, Level& fine);
};
