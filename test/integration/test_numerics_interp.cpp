// Numerics: interpolation (bi/trilinear MAC sampling), advection invariants,
// and finite-volume operators (divergence / laplacian / vorticity).
//
// CPU-only. Links against liblfm_lib.a (no GPU/CUDA symbols).
#include "../test_utils.h"

#include "core/grid.h"
#include "core/grid_3d.h"
#include "integrator/ops/advection.h"
#include "integrator/ops/advection_3d.h"
#include "numerics/ops/operators.h"
#include "numerics/ops/operators_3d.h"

#include <cmath>

// ── Fill a 2D MAC grid so that the face velocities sample an analytic
//    linear field  u(x,y) = a + bx + cy ,  v(x,y) = d + ex + fy.
//    u-face physical position: (i*dx, (j-0.5)*dy)
//    v-face physical position: ((i-0.5)*dx, j*dy)
static void fill_linear_2d(Grid& g, double a, double b, double c, double d, double e, double f) {
    for (int j = 0; j <= g.ny + 1; ++j)
        for (int i = 0; i <= g.nx; ++i) {
            double x        = i * g.dx;
            double y        = (j - 0.5) * g.dy;
            g.u[g.iu(i, j)] = a + b * x + c * y;
        }
    for (int j = 0; j <= g.ny; ++j)
        for (int i = 0; i <= g.nx + 1; ++i) {
            double x        = (i - 0.5) * g.dx;
            double y        = j * g.dy;
            g.v[g.iv(i, j)] = d + e * x + f * y;
        }
}

int main() {
    test_header("Numerics: interpolation, advection invariants, FV operators");

    const double EPS = 1e-9;

    // ===================================================================
    // 1. 2D bilinear MAC interpolation — node values
    // ===================================================================
    {
        Grid g(8, 6, 8.0, 6.0); // dx = dy = 1.0
        fill_linear_2d(g, 0.3, 2.0, -1.5, -0.7, 0.4, 1.1);

        // Sampling exactly at a u-face node returns the stored node value.
        for (int i = 1; i <= g.nx - 1; ++i)
            for (int j = 1; j <= g.ny; ++j) {
                double x   = i * g.dx;
                double y   = (j - 0.5) * g.dy;
                double got = AdvectionScheme::sampleU(g, x, y);
                check_approx(got, g.u_at(i, j), EPS,
                             "sampleU at u-node returns node value (i=" + std::to_string(i) +
                                 ",j=" + std::to_string(j) + ")");
            }
    }

    // ===================================================================
    // 2. 2D bilinear MAC interpolation — exact on a linear field
    // ===================================================================
    {
        Grid g(10, 10, 10.0, 10.0);
        const double a = 1.2, b = 0.5, c = -0.8;  // u = a + b x + c y
        const double d = -0.4, e = -0.9, f = 0.6; // v = d + e x + f y
        fill_linear_2d(g, a, b, c, d, e, f);

        // Arbitrary interior points (kept away from the domain edges so the
        // clamp in sampleU/sampleV never activates).
        double pts[][2] = {{3.27, 4.81}, {5.5, 5.5}, {2.0, 7.3}, {6.13, 1.97}, {4.0, 4.0}};
        for (auto& pt : pts) {
            double x      = pt[0], y = pt[1];
            double exactU = a + b * x + c * y;
            double exactV = d + e * x + f * y;
            check_approx(AdvectionScheme::sampleU(g, x, y), exactU, 1e-9,
                         "sampleU exact on linear field @(" + std::to_string(x) + "," +
                             std::to_string(y) + ")");
            check_approx(AdvectionScheme::sampleV(g, x, y), exactV, 1e-9,
                         "sampleV exact on linear field @(" + std::to_string(x) + "," +
                             std::to_string(y) + ")");
        }
    }

    // ===================================================================
    // 3. 2D advection invariants
    // ===================================================================
    {
        // Constant velocity field is invariant under self-advection: a uniform
        // field backtraces to a point where the sampled value equals itself.
        Grid g(8, 8, 8.0, 8.0);
        const double U = 1.3, V = -0.7;
        for (auto& val : g.u)
            val = U;
        for (auto& val : g.v)
            val = V;
        Grid ng = g;
        AdvectionScheme::advect(g, ng, 0.25);
        bool u_const = true, v_const = true;
        for (int i = 1; i < g.nx; ++i)
            for (int j = 1; j <= g.ny; ++j)
                if (std::abs(ng.u_at(i, j) - U) > 1e-9)
                    u_const = false;
        for (int i = 1; i <= g.nx; ++i)
            for (int j = 1; j < g.ny; ++j)
                if (std::abs(ng.v_at(i, j) - V) > 1e-9)
                    v_const = false;
        check(u_const, "advect: constant u-field invariant under self-advection");
        check(v_const, "advect: constant v-field invariant under self-advection");
    }
    {
        // dt = 0 leaves the field unchanged: backtrace returns the same node,
        // and sampling at the node returns the node value.
        Grid g(10, 10, 10.0, 10.0);
        fill_linear_2d(g, 0.5, 1.1, 0.3, -0.2, 0.7, -0.5);
        Grid ng = g;
        AdvectionScheme::advect(g, ng, 0.0);
        bool unchanged = true;
        for (int i = 1; i < g.nx; ++i)
            for (int j = 1; j <= g.ny; ++j)
                if (std::abs(ng.u_at(i, j) - g.u_at(i, j)) > 1e-9)
                    unchanged = false;
        for (int i = 1; i <= g.nx; ++i)
            for (int j = 1; j < g.ny; ++j)
                if (std::abs(ng.v_at(i, j) - g.v_at(i, j)) > 1e-9)
                    unchanged = false;
        check(unchanged, "advect: dt=0 leaves field unchanged");
    }
    {
        // Zero velocity ⇒ no transport ⇒ any field unchanged.
        Grid g(8, 8, 8.0, 8.0);
        fill_linear_2d(g, 2.0, -0.3, 0.9, 1.0, 0.2, -0.6);
        // Zero out the velocity *used for transport* by making it a separate
        // self-advection: here the transported field IS the velocity, so we
        // instead test the canonical "zero velocity" via a constant-zero grid.
        Grid z(8, 8, 8.0, 8.0); // all zero
        Grid nz = z;
        AdvectionScheme::advect(z, nz, 0.5);
        bool zero_stays_zero = true;
        for (int i = 1; i < z.nx; ++i)
            for (int j = 1; j <= z.ny; ++j)
                if (std::abs(nz.u_at(i, j)) > 1e-12)
                    zero_stays_zero = false;
        check(zero_stays_zero, "advect: zero velocity field stays zero");
    }

    // ===================================================================
    // 4. 2D finite-volume operators
    // ===================================================================
    {
        // Divergence exact on a linear field.
        // u = a + b x  ⇒ ∂u/∂x = b ;  v = d + f y ⇒ ∂v/∂y = f.
        Grid g(12, 12, 12.0, 12.0);
        const double b = 0.73, f = -0.41;
        fill_linear_2d(g, 0.1, b, 0.0, -0.2, 0.0, f);
        double expected = b + f;
        for (int i = 2; i <= g.nx - 1; ++i)
            for (int j = 2; j <= g.ny - 1; ++j)
                check_approx(fvc::divergence(g, i, j), expected, 1e-9,
                             "divergence(2D) exact on linear field");
    }
    {
        // Divergence-free field has zero divergence everywhere.
        Grid g(10, 10, 10.0, 10.0);
        // u = y, v = -x  ⇒ ∇·u = 0  (solid-body rotation).
        fill_linear_2d(g, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0);
        bool divfree = true;
        for (int i = 2; i <= g.nx - 1; ++i)
            for (int j = 2; j <= g.ny - 1; ++j)
                if (std::abs(fvc::divergence(g, i, j)) > 1e-9)
                    divfree = false;
        check(divfree, "divergence(2D) is zero on a divergence-free field");
    }
    {
        // Laplacian exact on a quadratic scalar field.
        // s = x^2 + y^2  ⇒ ∇²s = 4 (second differences are exact for quadratics).
        Grid g(12, 12, 12.0, 12.0);
        std::vector<double> s(g.p_size(), 0.0);
        for (int j = 0; j <= g.ny + 1; ++j)
            for (int i = 0; i <= g.nx + 1; ++i) {
                double x       = (i - 0.5) * g.dx;
                double y       = (j - 0.5) * g.dy;
                s[g.ip(i, j)]  = x * x + y * y;
            }
        for (int i = 2; i <= g.nx - 1; ++i)
            for (int j = 2; j <= g.ny - 1; ++j)
                check_approx(fvc::laplacian(g, s, i, j), 4.0, 1e-7,
                             "laplacian(2D) exact on s=x^2+y^2 (= 4)");
    }
    {
        // Laplacian of a linear field is zero.
        Grid g(10, 10, 10.0, 10.0);
        std::vector<double> s(g.p_size(), 0.0);
        for (int j = 0; j <= g.ny + 1; ++j)
            for (int i = 0; i <= g.nx + 1; ++i) {
                double x      = (i - 0.5) * g.dx;
                double y      = (j - 0.5) * g.dy;
                s[g.ip(i, j)] = 3.0 + 2.0 * x - 1.5 * y;
            }
        bool zero = true;
        for (int i = 2; i <= g.nx - 1; ++i)
            for (int j = 2; j <= g.ny - 1; ++j)
                if (std::abs(fvc::laplacian(g, s, i, j)) > 1e-8)
                    zero = false;
        check(zero, "laplacian(2D) of a linear field is zero");
    }
    {
        // Vorticity of solid-body rotation u=-y, v=x is constant ω = 2.
        // (∂v/∂x − ∂u/∂y = 1 − (−1) = 2.)
        Grid g(12, 12, 12.0, 12.0);
        fill_linear_2d(g, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        for (int i = 2; i <= g.nx - 1; ++i)
            for (int j = 2; j <= g.ny - 1; ++j)
                check_approx(fvc::vorticity(g, i, j), 2.0, 1e-9,
                             "vorticity(2D) of solid-body rotation = 2");
        // Irrotational (uniform) flow has zero vorticity.
        Grid u(8, 8, 8.0, 8.0);
        for (auto& val : u.u)
            val = 2.5;
        for (auto& val : u.v)
            val = -1.0;
        check_approx(fvc::vorticity(u, 4, 4), 0.0, 1e-9,
                     "vorticity(2D) of uniform flow is zero");
    }

    // ===================================================================
    // 5. 3D trilinear interpolation — node values + exact on linear field
    // ===================================================================
    {
        Grid3D g(6, 6, 6, 6.0, 6.0, 6.0); // dx=dy=dz=1
        // u(x,y,z) = a + b x + c y + e z
        const double a = 0.4, b = 0.9, c = -0.5, e = 0.3;
        for (int k = 0; k <= g.nz + 1; ++k)
            for (int j = 0; j <= g.ny + 1; ++j)
                for (int i = 0; i <= g.nx; ++i) {
                    double x           = i * g.dx;
                    double y           = (j - 0.5) * g.dy;
                    double z           = (k - 0.5) * g.dz;
                    g.u[g.iu(i, j, k)] = a + b * x + c * y + e * z;
                }
        // Node values.
        for (int i = 1; i <= g.nx - 1; ++i)
            check_approx(AdvectionScheme3D::sampleU(g, i * g.dx, (3 - 0.5) * g.dy, (3 - 0.5) * g.dz),
                         g.u_at(i, 3, 3), 1e-9, "sampleU(3D) at node returns node value");
        // Exact on linear field at arbitrary interior points.
        double pts[][3] = {{2.3, 3.1, 4.4}, {4.0, 4.0, 4.0}, {1.7, 2.9, 3.3}};
        for (auto& p : pts) {
            double exact = a + b * p[0] + c * p[1] + e * p[2];
            check_approx(AdvectionScheme3D::sampleU(g, p[0], p[1], p[2]), exact, 1e-9,
                         "sampleU(3D) exact on linear field");
        }
    }

    // ===================================================================
    // 6. 3D advection invariants + FV operators
    // ===================================================================
    {
        // Constant velocity invariant under self-advection.
        Grid3D g(6, 6, 6, 6.0, 6.0, 6.0);
        const double U = 0.8, V = -0.5, W = 0.3;
        for (auto& val : g.u)
            val = U;
        for (auto& val : g.v)
            val = V;
        for (auto& val : g.w)
            val = W;
        Grid3D ng = g;
        AdvectionScheme3D::advect(g, ng, 0.2);
        bool ok = true;
        for (int i = 1; i < g.nx; ++i)
            for (int j = 1; j <= g.ny; ++j)
                for (int k = 1; k <= g.nz; ++k)
                    if (std::abs(ng.u_at(i, j, k) - U) > 1e-9)
                        ok = false;
        check(ok, "advect(3D): constant u-field invariant under self-advection");
    }
    {
        // 3D divergence exact on a linear field: u=b x, v=f y, w=h z ⇒ ∇·u = b+f+h.
        Grid3D g(8, 8, 8, 8.0, 8.0, 8.0);
        const double b = 0.5, f = -0.2, h = 0.7;
        for (int k = 0; k <= g.nz + 1; ++k)
            for (int j = 0; j <= g.ny + 1; ++j)
                for (int i = 0; i <= g.nx; ++i)
                    g.u[g.iu(i, j, k)] = b * (i * g.dx);
        for (int k = 0; k <= g.nz + 1; ++k)
            for (int j = 0; j <= g.ny; ++j)
                for (int i = 0; i <= g.nx + 1; ++i)
                    g.v[g.iv(i, j, k)] = f * (j * g.dy);
        for (int k = 0; k <= g.nz; ++k)
            for (int j = 0; j <= g.ny + 1; ++j)
                for (int i = 0; i <= g.nx + 1; ++i)
                    g.w[g.iw(i, j, k)] = h * (k * g.dz);
        check_approx(fvc::divergence(g, 4, 4, 4), b + f + h, 1e-9,
                     "divergence(3D) exact on linear field");
    }
    {
        // 3D laplacian exact on s = x^2 + y^2 + z^2 ⇒ ∇²s = 6.
        Grid3D g(8, 8, 8, 8.0, 8.0, 8.0);
        std::vector<double> s(g.p_size(), 0.0);
        for (int k = 0; k <= g.nz + 1; ++k)
            for (int j = 0; j <= g.ny + 1; ++j)
                for (int i = 0; i <= g.nx + 1; ++i) {
                    double x          = (i - 0.5) * g.dx;
                    double y          = (j - 0.5) * g.dy;
                    double z          = (k - 0.5) * g.dz;
                    s[g.ip(i, j, k)]  = x * x + y * y + z * z;
                }
        check_approx(fvc::laplacian(g, s, 4, 4, 4), 6.0, 1e-7,
                     "laplacian(3D) exact on s=x^2+y^2+z^2 (= 6)");
    }
    {
        // 3D vorticity of planar solid-body rotation (u=-y, v=x, w=0) ⇒ ω=(0,0,2).
        Grid3D g(8, 8, 8, 8.0, 8.0, 8.0);
        for (int k = 0; k <= g.nz + 1; ++k)
            for (int j = 0; j <= g.ny + 1; ++j)
                for (int i = 0; i <= g.nx; ++i) {
                    double y           = (j - 0.5) * g.dy;
                    g.u[g.iu(i, j, k)] = -y;
                }
        for (int k = 0; k <= g.nz + 1; ++k)
            for (int j = 0; j <= g.ny; ++j)
                for (int i = 0; i <= g.nx + 1; ++i) {
                    double x           = (i - 0.5) * g.dx;
                    g.v[g.iv(i, j, k)] = x;
                }
        auto w = fvc::vorticity(g, 4, 4, 4);
        check_approx(w[0], 0.0, 1e-9, "vorticity(3D) x-comp of planar rotation = 0");
        check_approx(w[1], 0.0, 1e-9, "vorticity(3D) y-comp of planar rotation = 0");
        check_approx(w[2], 2.0, 1e-9, "vorticity(3D) z-comp of planar rotation = 2");
        check_approx(fvc::vorticity_magnitude(g, 4, 4, 4), 2.0, 1e-9,
                     "vorticity_magnitude(3D) of planar rotation = 2");
    }

    return test_summary();
}
