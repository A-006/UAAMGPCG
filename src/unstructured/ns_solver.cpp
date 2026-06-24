/**
 * @file ns_solver.cpp
 * @brief Chorin fractional-step incompressible NS on a collocated FVM mesh.
 * @author liutao
 * @date 2026-06-23
 */
#include "unstructured/ns_solver.h"
#include "unstructured/fvm_poisson.h"

#include <cmath>

namespace ufvm {
namespace {
inline double g_diff(const Face& f, Vec2 d) {
    return dot(f.Sf, f.Sf) / dot(f.Sf, d);
}
inline Vec2 face_dvec(const PolyMesh& m, const Face& f) {
    return (f.nb >= 0 ? m.centroid[f.nb] : f.Cf) - m.centroid[f.owner];
}
} // namespace

NSState ns_init(const PolyMesh& m, const SpaceTime& u0, const SpaceTime& v0, const SpaceTime& p0,
                double t0) {
    NSState s;
    const int n = m.n_cells;
    s.u.resize(n);
    s.v.resize(n);
    s.p.resize(n);
    for (int c = 0; c < n; ++c) {
        s.u[c] = u0(m.centroid[c], t0);
        s.v[c] = v0(m.centroid[c], t0);
        s.p[c] = p0(m.centroid[c], t0);
    }
    s.faceFlux.resize(m.faces.size());
    for (size_t fi = 0; fi < m.faces.size(); ++fi) {
        const Face& f = m.faces[fi];
        Vec2 uf;
        if (f.nb >= 0)
            uf = {0.5 * (s.u[f.owner] + s.u[f.nb]), 0.5 * (s.v[f.owner] + s.v[f.nb])};
        else
            uf = {u0(f.Cf, t0), v0(f.Cf, t0)};
        s.faceFlux[fi] = dot(uf, f.Sf);
    }
    return s;
}

void ns_step(const PolyMesh& m, const CSR& A, NSState& s, const NSBoundary& bc, const NSConfig& cfg,
             double t, double dt, const std::function<std::vector<double>(const CSR&)>& solver) {
    const int n  = m.n_cells;
    const int nf = static_cast<int>(m.faces.size());
    auto bcu     = [&](Vec2 x) { return bc.bc_u(x, t); };
    auto bcv     = [&](Vec2 x) { return bc.bc_v(x, t); };
    auto bcp     = [&](Vec2 x) { return bc.bc_p(x, t + dt); };

    // velocity gradients (for non-orthogonal diffusion correction)
    std::vector<Vec2> gu = ls_gradient(m, s.u, bcu);
    std::vector<Vec2> gv = ls_gradient(m, s.v, bcv);

    // ---- momentum predictor: -convection + nu*diffusion (explicit) ----
    std::vector<double> rU(n, 0.0), rV(n, 0.0); // accumulated face contributions
    for (int fi = 0; fi < nf; ++fi) {
        const Face& f = m.faces[fi];
        const int P = f.owner, N = f.nb;
        Vec2 d    = face_dvec(m, f);
        double gd = g_diff(f, d);
        Vec2 Tf   = {f.Sf.x - gd * d.x, f.Sf.y - gd * d.y};
        double F  = s.faceFlux[fi];
        if (N >= 0) {
            double uf = 0.5 * (s.u[P] + s.u[N]), vf = 0.5 * (s.v[P] + s.v[N]);
            Vec2 gfu  = {0.5 * (gu[P].x + gu[N].x), 0.5 * (gu[P].y + gu[N].y)};
            Vec2 gfv  = {0.5 * (gv[P].x + gv[N].x), 0.5 * (gv[P].y + gv[N].y)};
            double dU = gd * (s.u[N] - s.u[P]) + dot(gfu, Tf); // viscous flux owner->nb
            double dV = gd * (s.v[N] - s.v[P]) + dot(gfv, Tf);
            double cU = F * uf, cV = F * vf; // convective flux
            rU[P] += -cU + cfg.nu * dU;
            rU[N] -= -cU + cfg.nu * dU;
            rV[P] += -cV + cfg.nu * dV;
            rV[N] -= -cV + cfg.nu * dV;
        } else {
            double uf = bcu(f.Cf), vf = bcv(f.Cf);
            double dU = gd * (uf - s.u[P]) + dot(gu[P], Tf);
            double dV = gd * (vf - s.v[P]) + dot(gv[P], Tf);
            double cU = F * uf, cV = F * vf;
            rU[P] += -cU + cfg.nu * dU;
            rV[P] += -cV + cfg.nu * dV;
        }
    }
    std::vector<double> us(n), vs(n);
    for (int c = 0; c < n; ++c) {
        us[c] = s.u[c] + dt / m.vol[c] * rU[c];
        vs[c] = s.v[c] + dt / m.vol[c] * rV[c];
    }

    // ---- predictor face mass fluxes + discrete divergence ----
    std::vector<double> Fstar(nf), divc(n, 0.0);
    for (int fi = 0; fi < nf; ++fi) {
        const Face& f = m.faces[fi];
        const int P = f.owner, N = f.nb;
        Vec2 uf;
        if (N >= 0)
            uf = {0.5 * (us[P] + us[N]), 0.5 * (vs[P] + vs[N])};
        else
            uf = {bcu(f.Cf), bcv(f.Cf)};
        Fstar[fi] = dot(uf, f.Sf);
        divc[P] += Fstar[fi];
        if (N >= 0)
            divc[N] -= Fstar[fi];
    }

    // ---- pressure Poisson: A p = b,  ∇²p = div(u*)/dt  (deferred non-orth) ----
    CSR Ap = A;
    std::vector<double> p(n, 0.0);
    std::vector<Vec2> gp(n, {0, 0});
    for (int outer = 0; outer < cfg.p_outer; ++outer) {
        std::vector<double> b(n, 0.0);
        for (int c = 0; c < n; ++c)
            b[c] = -divc[c] / dt; // -∫f dV  (divc already volume-integrated)
        for (int fi = 0; fi < nf; ++fi) {
            const Face& f = m.faces[fi];
            const int P = f.owner, N = f.nb;
            Vec2 d    = face_dvec(m, f);
            double gd = g_diff(f, d);
            Vec2 Tf   = {f.Sf.x - gd * d.x, f.Sf.y - gd * d.y};
            if (N >= 0) {
                Vec2 gf     = {0.5 * (gp[P].x + gp[N].x), 0.5 * (gp[P].y + gp[N].y)};
                double corr = dot(gf, Tf);
                b[P] += corr;
                b[N] -= corr;
            } else {
                b[P] += gd * bcp(f.Cf) + dot(gp[P], Tf);
            }
        }
        Ap.b = b;
        p    = solver(Ap);
        gp   = ls_gradient(m, p, bcp);
    }
    s.p = p;

    // ---- velocity & face-flux correction ----
    for (int c = 0; c < n; ++c) {
        s.u[c] = us[c] - dt * gp[c].x;
        s.v[c] = vs[c] - dt * gp[c].y;
    }
    for (int fi = 0; fi < nf; ++fi) {
        const Face& f = m.faces[fi];
        const int P = f.owner, N = f.nb;
        if (N >= 0) {
            double gd      = g_diff(f, face_dvec(m, f));
            s.faceFlux[fi] = Fstar[fi] - dt * gd * (p[N] - p[P]);
        } else {
            s.faceFlux[fi] = Fstar[fi]; // velocity-Dirichlet: flux set by BC
        }
    }
}

} // namespace ufvm
