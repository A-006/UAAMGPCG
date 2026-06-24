/**
 * @file fvm_poisson.cpp
 * @brief Implementation of the unstructured finite-volume Poisson assembly.
 * @author liutao
 * @date 2026-06-23
 */
#include "unstructured/fvm_poisson.h"

#include <cmath>
#include <map>

namespace ufvm {

namespace {
// over-relaxed orthogonal coefficient for a face, given owner->face vector d
inline double g_diff(const Face& f, Vec2 d) {
    return dot(f.Sf, f.Sf) / dot(f.Sf, d);
}
inline Vec2 face_dvec(const PolyMesh& m, const Face& f) {
    return (f.nb >= 0 ? m.centroid[f.nb] : f.Cf) - m.centroid[f.owner];
}
} // namespace

CSR assemble_laplacian(const PolyMesh& m, const ScalarField&) {
    const int n = m.n_cells;
    std::vector<std::map<int, double>> rows(n);
    for (int c = 0; c < n; ++c)
        rows[c][c] += 0.0; // ensure diagonal present

    for (const auto& f : m.faces) {
        const int P = f.owner, N = f.nb;
        const double gd = g_diff(f, face_dvec(m, f));
        if (N >= 0) {
            rows[P][P] += gd;
            rows[P][N] -= gd;
            rows[N][N] += gd;
            rows[N][P] -= gd;
        } else {
            rows[P][P] += gd; // Dirichlet face: known value moved to RHS
        }
    }

    CSR A;
    A.n = n;
    A.row_ptr.reserve(n + 1);
    A.row_ptr.push_back(0);
    for (int c = 0; c < n; ++c) {
        for (const auto& e : rows[c]) { // std::map keeps columns sorted
            A.col_idx.push_back(e.first);
            A.vals.push_back(e.second);
        }
        A.row_ptr.push_back(static_cast<int>(A.col_idx.size()));
    }
    A.b.assign(n, 0.0);
    return A;
}

CSR assemble_laplacian(const PolyMesh& m, const ScalarField&, const BCTypeField& bctype) {
    const int n = m.n_cells;
    std::vector<std::map<int, double>> rows(n);
    for (int c = 0; c < n; ++c)
        rows[c][c] += 0.0;

    for (const auto& f : m.faces) {
        const int P = f.owner, N = f.nb;
        const double gd = g_diff(f, face_dvec(m, f));
        if (N >= 0) {
            rows[P][P] += gd;
            rows[P][N] -= gd;
            rows[N][N] += gd;
            rows[N][P] -= gd;
        } else if (bctype(f.Cf) == BCType::Dirichlet) {
            rows[P][P] += gd; // Dirichlet: known value -> RHS
        }
        // Neumann (zero-gradient) boundary face: zero flux, contributes nothing.
    }

    CSR A;
    A.n = n;
    A.row_ptr.reserve(n + 1);
    A.row_ptr.push_back(0);
    for (int c = 0; c < n; ++c) {
        for (const auto& e : rows[c]) {
            A.col_idx.push_back(e.first);
            A.vals.push_back(e.second);
        }
        A.row_ptr.push_back(static_cast<int>(A.col_idx.size()));
    }
    A.b.assign(n, 0.0);
    return A;
}

std::vector<Vec2> ls_gradient(const PolyMesh& m, const std::vector<double>& p,
                              const ScalarField& dirichlet, const BCTypeField& bctype) {
    const int n = m.n_cells;
    std::vector<double> a11(n, 0), a12(n, 0), a22(n, 0), b1(n, 0), b2(n, 0);
    auto add = [&](int c, Vec2 d, double dp) {
        double w = 1.0 / dot(d, d);
        a11[c] += w * d.x * d.x;
        a12[c] += w * d.x * d.y;
        a22[c] += w * d.y * d.y;
        b1[c] += w * d.x * dp;
        b2[c] += w * d.y * dp;
    };
    for (const auto& f : m.faces) {
        if (f.nb >= 0) {
            Vec2 d    = m.centroid[f.nb] - m.centroid[f.owner];
            double dp = p[f.nb] - p[f.owner];
            add(f.owner, d, dp);
            add(f.nb, {-d.x, -d.y}, -dp);
        } else if (bctype(f.Cf) == BCType::Dirichlet) {
            Vec2 d = f.Cf - m.centroid[f.owner];
            add(f.owner, d, dirichlet(f.Cf) - p[f.owner]);
        }
        // Neumann face: omit from the LS fit (forcing p_f=p_P would be only
        // 1st-order); the interior-neighbour rows determine the gradient.
    }
    std::vector<Vec2> g(n);
    for (int c = 0; c < n; ++c) {
        double det = a11[c] * a22[c] - a12[c] * a12[c];
        g[c] = {(a22[c] * b1[c] - a12[c] * b2[c]) / det, (a11[c] * b2[c] - a12[c] * b1[c]) / det};
    }
    return g;
}

std::vector<double> build_rhs(const PolyMesh& m, const ScalarField& f, const ScalarField& dirichlet,
                              const std::vector<Vec2>& grad, const BCTypeField& bctype) {
    const int n = m.n_cells;
    std::vector<double> b(n, 0.0);
    for (int c = 0; c < n; ++c)
        b[c] = -f(m.centroid[c]) * m.vol[c];

    for (const auto& face : m.faces) {
        const int P = face.owner, N = face.nb;
        Vec2 d    = face_dvec(m, face);
        double gd = g_diff(face, d);
        Vec2 Tf{face.Sf.x - gd * d.x, face.Sf.y - gd * d.y};
        Vec2 gf =
            (N >= 0) ? Vec2{0.5 * (grad[P].x + grad[N].x), 0.5 * (grad[P].y + grad[N].y)} : grad[P];
        double corr = dot(gf, Tf);
        if (N >= 0) {
            b[P] += corr;
            b[N] -= corr;
        } else if (bctype(face.Cf) == BCType::Dirichlet) {
            b[P] += gd * dirichlet(face.Cf) + corr;
        }
        // Neumann face: zero flux -> no RHS contribution.
    }
    return b;
}

std::vector<double> solve_poisson(const PolyMesh& m, const ScalarField& f,
                                  const ScalarField& dirichlet, int n_outer,
                                  const std::function<std::vector<double>(const CSR&)>& solver,
                                  const BCTypeField& bctype) {
    CSR A = assemble_laplacian(m, dirichlet, bctype);
    std::vector<double> p(m.n_cells, 0.0);
    std::vector<Vec2> grad(m.n_cells, {0, 0});
    for (int outer = 0; outer < n_outer; ++outer) {
        A.b  = build_rhs(m, f, dirichlet, grad, bctype);
        p    = solver(A);
        grad = ls_gradient(m, p, dirichlet, bctype);
    }
    return p;
}

std::vector<Vec2> ls_gradient(const PolyMesh& m, const std::vector<double>& p,
                              const ScalarField& dirichlet) {
    const int n = m.n_cells;
    std::vector<double> a11(n, 0), a12(n, 0), a22(n, 0), b1(n, 0), b2(n, 0);
    auto add = [&](int c, Vec2 d, double dp) {
        double w = 1.0 / dot(d, d);
        a11[c] += w * d.x * d.x;
        a12[c] += w * d.x * d.y;
        a22[c] += w * d.y * d.y;
        b1[c] += w * d.x * dp;
        b2[c] += w * d.y * dp;
    };
    for (const auto& f : m.faces) {
        if (f.nb >= 0) {
            Vec2 d    = m.centroid[f.nb] - m.centroid[f.owner];
            double dp = p[f.nb] - p[f.owner];
            add(f.owner, d, dp);
            add(f.nb, {-d.x, -d.y}, -dp);
        } else {
            Vec2 d = f.Cf - m.centroid[f.owner];
            add(f.owner, d, dirichlet(f.Cf) - p[f.owner]);
        }
    }
    std::vector<Vec2> g(n);
    for (int c = 0; c < n; ++c) {
        double det = a11[c] * a22[c] - a12[c] * a12[c];
        g[c] = {(a22[c] * b1[c] - a12[c] * b2[c]) / det, (a11[c] * b2[c] - a12[c] * b1[c]) / det};
    }
    return g;
}

std::vector<double> build_rhs(const PolyMesh& m, const ScalarField& f, const ScalarField& dirichlet,
                              const std::vector<Vec2>& grad) {
    const int n = m.n_cells;
    std::vector<double> b(n, 0.0);
    for (int c = 0; c < n; ++c)
        b[c] = -f(m.centroid[c]) * m.vol[c]; // -∫f dV

    for (const auto& face : m.faces) {
        const int P = face.owner, N = face.nb;
        Vec2 d    = face_dvec(m, face);
        double gd = g_diff(face, d);
        Vec2 Tf{face.Sf.x - gd * d.x, face.Sf.y - gd * d.y}; // non-orthogonal part
        Vec2 gf =
            (N >= 0) ? Vec2{0.5 * (grad[P].x + grad[N].x), 0.5 * (grad[P].y + grad[N].y)} : grad[P];
        double corr = dot(gf, Tf);
        if (N >= 0) {
            b[P] += corr;
            b[N] -= corr;
        } else {
            b[P] += gd * dirichlet(face.Cf) + corr;
        }
    }
    return b;
}

std::vector<double> matvec(const CSR& A, const std::vector<double>& x) {
    std::vector<double> y(A.n, 0.0);
    for (int i = 0; i < A.n; ++i) {
        double s = 0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            s += A.vals[k] * x[A.col_idx[k]];
        y[i] = s;
    }
    return y;
}

std::vector<double> cg_solve(const CSR& A, const std::vector<double>& b, int max_iter, double tol) {
    const int n = A.n;
    std::vector<double> x(n, 0.0), r = b, p = b, Ap(n);
    double rr = 0;
    for (double v : r)
        rr += v * v;
    double rr0 = rr;
    for (int it = 0; it < max_iter && rr > tol * tol * rr0; ++it) {
        Ap         = matvec(A, p);
        double pAp = 0;
        for (int i = 0; i < n; ++i)
            pAp += p[i] * Ap[i];
        double alpha = rr / pAp;
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        double rr1 = 0;
        for (double v : r)
            rr1 += v * v;
        double beta = rr1 / rr;
        for (int i = 0; i < n; ++i)
            p[i] = r[i] + beta * p[i];
        rr = rr1;
    }
    return x;
}

std::vector<double> solve_poisson(const PolyMesh& m, const ScalarField& f,
                                  const ScalarField& dirichlet, int n_outer,
                                  const std::function<std::vector<double>(const CSR&)>& solver) {
    CSR A = assemble_laplacian(m, dirichlet);
    std::vector<double> p(m.n_cells, 0.0);
    std::vector<Vec2> grad(m.n_cells, {0, 0});
    for (int outer = 0; outer < n_outer; ++outer) {
        A.b  = build_rhs(m, f, dirichlet, grad); // refresh deferred correction
        p    = solver(A);
        grad = ls_gradient(m, p, dirichlet);
    }
    return p;
}

} // namespace ufvm
