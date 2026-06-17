#include "../test_utils.h"
#include "core/grid_3d.h"
#include "core/scalar_field_3d.h"
#include <cmath>

// Unit tests for ScalarField3D: construction, fill, indexing,
// semi-Lagrangian advection invariants, and Boussinesq buoyancy.

int main() {
    test_header("ScalarField3D — fill, advect invariants, buoyancy");

    int nx = 8, ny = 8, nz = 8;
    double L = 8.0; // dx=dy=dz=1
    Grid3D g(nx, ny, nz, L, L, L);

    // ── Construction: filled with 0 ─────────────────────────────
    {
        ScalarField3D s(g);
        check((int)s.data().size() == g.p_size(), "ScalarField3D sized to p_size()");
        bool all_zero = true;
        for (double x : s.data()) all_zero &= (x == 0.0);
        check(all_zero, "ScalarField3D constructed filled with 0");
    }

    // ── fill() and operator()(i,j,k) ────────────────────────────
    {
        ScalarField3D s(g);
        s.fill(2.75);
        bool all = true;
        for (double x : s.data()) all &= (x == 2.75);
        check(all, "fill() sets every entry");
        check_approx(s(3, 4, 5), 2.75, 1e-12, "operator() reads filled value");

        s(3, 4, 5) = -1.5;
        check_approx(s(3, 4, 5), -1.5, 1e-12, "operator() write/read round-trip");
        check_approx(s.data()[g.ip(3, 4, 5)], -1.5, 1e-12, "operator() writes ip()-indexed slot");
        // The const overload sees the same value.
        const ScalarField3D& cs = s;
        check_approx(cs(3, 4, 5), -1.5, 1e-12, "const operator() reads same value");
    }

    // ── advect invariant 1: constant scalar stays constant ──────
    // A spatially-constant field advected by ANY velocity must remain constant
    // (interior cells). Set a non-trivial divergent velocity field.
    {
        Grid3D flow(nx, ny, nz, L, L, L);
        for (auto& x : flow.u) x = 0.7;
        for (auto& x : flow.v) x = -0.4;
        for (auto& x : flow.w) x = 0.9;
        // Make it spatially varying too, to stress the sampler.
        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++)
                for (int i = 0; i <= nx; i++)
                    flow.u_at(i, j, k) = 0.3 * i - 0.1 * j + 0.05 * k;

        ScalarField3D src(g), dst(g);
        src.fill(3.14159);
        dst.fill(-999.0); // poison, must be overwritten
        ScalarField3D::advect(src, dst, flow, 0.5);

        bool constant = true;
        double maxerr = 0.0;
        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++)
                for (int i = 1; i <= nx; i++) {
                    double e = std::abs(dst(i, j, k) - 3.14159);
                    maxerr = std::max(maxerr, e);
                    constant &= (e < 1e-9);
                }
        check(constant, "constant scalar advected by varying velocity stays constant");
        check_approx(maxerr, 0.0, 1e-9, "max deviation of advected constant ~ 0");
    }

    // ── advect invariant 2: zero velocity leaves field unchanged ─
    {
        Grid3D still(nx, ny, nz, L, L, L); // all-zero velocity by construction
        ScalarField3D src(g), dst(g);
        // Arbitrary spatial pattern.
        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++)
                for (int i = 1; i <= nx; i++)
                    src(i, j, k) = std::sin(0.5 * i) + 0.3 * j - 0.2 * k;
        ScalarField3D::advect(src, dst, still, 0.25);

        bool unchanged = true;
        double maxerr = 0.0;
        for (int k = 1; k <= nz; k++)
            for (int j = 1; j <= ny; j++)
                for (int i = 1; i <= nx; i++) {
                    double e = std::abs(dst(i, j, k) - src(i, j, k));
                    maxerr = std::max(maxerr, e);
                    unchanged &= (e < 1e-9);
                }
        check(unchanged, "zero velocity advection leaves field unchanged");
        check_approx(maxerr, 0.0, 1e-9, "max change under zero-velocity advect ~ 0");
    }

    // ── advect: linear field shifts by -u*dt (uniform velocity) ──
    // T(x)=x. Backtrace from cell center x by uniform u over dt lands at
    // x-u*dt, so the advected value at that center ~ x - u*dt. Generous
    // tolerance: RK2 semi-Lagrangian + trilinear + boundary clamping.
    {
        Grid3D flow(nx, ny, nz, L, L, L);
        double u0 = 1.0;
        double dt = 0.5;
        for (auto& x : flow.u) x = u0;
        // v, w stay zero.

        ScalarField3D src(g), dst(g);
        for (int k = 0; k <= nz + 1; k++)
            for (int j = 0; j <= ny + 1; j++)
                for (int i = 0; i <= nx + 1; i++)
                    src(i, j, k) = g.cell_x(i); // linear in x
        ScalarField3D::advect(src, dst, flow, dt);

        // Check interior cells away from the x-boundaries (clamping there).
        double maxerr = 0.0;
        for (int k = 2; k <= nz - 1; k++)
            for (int j = 2; j <= ny - 1; j++)
                for (int i = 3; i <= nx - 1; i++) {
                    double expected = g.cell_x(i) - u0 * dt;
                    maxerr = std::max(maxerr, std::abs(dst(i, j, k) - expected));
                }
        check(maxerr < 1e-6, "linear field shifts by -u*dt under uniform velocity");

        // Spot-check one interior cell explicitly with generous tolerance.
        int ci = 5, cj = 4, ck = 4;
        check_approx(dst(ci, cj, ck), g.cell_x(ci) - u0 * dt, 1e-3,
                     "spot cell shifted ~ x - u*dt");
    }

    // ── apply_buoyancy: T == T_ref everywhere -> velocity unchanged
    {
        Grid3D h(nx, ny, nz, L, L, L);
        // Seed w with a recognizable pattern.
        for (size_t n = 0; n < h.w.size(); n++) h.w[n] = 0.123 + 0.001 * (double)n;
        std::vector<double> w_before = h.w;
        std::vector<double> u_before = h.u, v_before = h.v;

        ScalarField3D T(h);
        double T_ref = 1.5;
        T.fill(T_ref);
        apply_buoyancy(h, T, T_ref, /*beta=*/2.0, /*dt=*/0.3);

        check(h.w == w_before, "T==T_ref leaves w unchanged");
        check(h.u == u_before && h.v == v_before, "buoyancy never touches u/v");
    }

    // ── apply_buoyancy: T > T_ref adds upward (w) force, exact increment
    // Source: w_at(i,j,k) += dt*beta*(T_face - T_ref),
    // T_face = 0.5*(T(i,j,k)+T(i,j,k+1)). Upward = +w. Loop k in [1, nz).
    {
        Grid3D h(nx, ny, nz, L, L, L);
        ScalarField3D T(h);
        double T_ref = 0.0, beta = 0.5, dt = 0.2;

        // Uniform hot field: T = 4 everywhere -> T_face = 4 on all w-faces.
        T.fill(4.0);
        apply_buoyancy(h, T, T_ref, beta, dt);

        double inc = dt * beta * (4.0 - T_ref); // = 0.2*0.5*4 = 0.4
        // Interior w-face (i,j,k) with k in [1, nz) gets the increment;
        // adjacent cells are non-solid.
        int ci = 4, cj = 4, ck = 3;
        check_approx(h.w_at(ci, cj, ck), inc, 1e-12,
                     "hot uniform field: exact upward increment dt*beta*(T-T_ref)");
        check(h.w_at(ci, cj, ck) > 0.0, "T>T_ref pushes w upward (positive)");

        int ci2 = 6, cj2 = 2, ck2 = 5;
        check_approx(h.w_at(ci2, cj2, ck2), inc, 1e-12, "second hot cell: same exact increment");

        // The k = nz w-face is NOT touched (loop is k < nz).
        check_approx(h.w_at(ci, cj, nz), 0.0, 1e-12, "top w-face (k=nz) untouched by buoyancy");
    }

    // ── apply_buoyancy: non-uniform T uses the 2-cell face average ──
    {
        Grid3D h(nx, ny, nz, L, L, L);
        ScalarField3D T(h);
        double T_ref = 1.0, beta = 3.0, dt = 0.1;
        T.fill(T_ref); // baseline -> zero contribution everywhere
        int ci = 3, cj = 3, ck = 4;
        // Make the two cells bracketing w-face (ci,cj,ck) hot/cold:
        T(ci, cj, ck)     = 5.0;
        T(ci, cj, ck + 1) = 3.0;
        apply_buoyancy(h, T, T_ref, beta, dt);

        double T_face = 0.5 * (5.0 + 3.0);              // = 4.0
        double inc = dt * beta * (T_face - T_ref);      // 0.1*3*(4-1)=0.9
        check_approx(h.w_at(ci, cj, ck), inc, 1e-12,
                     "non-uniform T: increment uses 0.5*(T(k)+T(k+1))");
        // A w-face whose two cells are both at T_ref gets nothing.
        check_approx(h.w_at(0 + 1, 0 + 1, 1), 0.0, 1e-12, "T==T_ref face: no buoyancy increment");
    }

    // ── apply_buoyancy: T < T_ref gives downward (negative) force ─
    {
        Grid3D h(nx, ny, nz, L, L, L);
        ScalarField3D T(h);
        double T_ref = 5.0, beta = 1.0, dt = 0.5;
        T.fill(2.0); // colder than ref everywhere
        apply_buoyancy(h, T, T_ref, beta, dt);
        double inc = dt * beta * (2.0 - T_ref); // negative
        check(inc < 0.0 && h.w_at(4, 4, 3) < 0.0, "T<T_ref pushes w downward (negative)");
        check_approx(h.w_at(4, 4, 3), inc, 1e-12, "cold field: exact negative increment");
    }

    return test_summary();
}
