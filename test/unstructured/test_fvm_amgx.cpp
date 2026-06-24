/**
 * @file test_fvm_amgx.cpp
 * @brief AmgX solves the unstructured FVM matrix — manufactured linear solve.
 *
 * Builds the FVM Laplacian on a triangle mesh, picks a known x_true, forms
 * b = A x_true, and solves via AmgxBackend with an FGMRES+classical-AMG config
 * (the robust choice for general unstructured pressure matrices). Asserts AmgX
 * recovers x_true and drives the residual down — proving the AmgxBackend reuse
 * is numerically sound on an unstructured matrix. Only built when AMGX is on.
 */
#include "../test_utils.h"
#include "solver/amgx_backend.h"
#include "unstructured/fvm_poisson.h"

#include <cmath>
#include <cstdio>

using namespace ufvm;

// FGMRES + classical AMG: robust for (possibly non-symmetric) unstructured matrices.
static const char* kFgmresCfg =
    "{\"config_version\":2,\"solver\":{"
    "\"solver\":\"FGMRES\",\"gmres_n_restart\":50,"
    "\"preconditioner\":{\"solver\":\"AMG\",\"algorithm\":\"CLASSICAL\",\"max_iters\":1,"
    "\"cycle\":\"V\",\"presweeps\":1,\"postsweeps\":1,\"aggressive_levels\":2,"
    "\"interpolator\":\"D2\",\"max_levels\":50,\"smoother\":{\"solver\":\"BLOCK_JACOBI\"}},"
    "\"max_iters\":300,\"tolerance\":1e-10,\"convergence\":\"RELATIVE_INI\",\"norm\":\"L2\","
    "\"monitor_residual\":1,\"obtain_timings\":0,\"print_solve_stats\":0}}";

int main() {
    test_header("Unstructured FVM matrix — AmgX (FGMRES) manufactured solve");

    PolyMesh m  = make_rect_tri(48);
    auto bc     = [](Vec2) { return 0.0; };
    CSR A       = assemble_laplacian(m, bc);
    const int n = A.n;

    std::vector<double> xtrue(n);
    for (int i = 0; i < n; ++i)
        xtrue[i] = std::sin(0.013 * i) + 0.5 * std::cos(0.007 * i);
    std::vector<double> b = matvec(A, xtrue);

    AmgxBackend amgx;
    amgx.set_config(kFgmresCfg);
    std::vector<double> x = amgx.solve_csr(n, A.row_ptr, A.col_idx, A.vals, b, 300, 1e-10);

    // recovery error and residual
    double en = 0, tn = 0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - xtrue[i];
        en += d * d;
        tn += xtrue[i] * xtrue[i];
    }
    std::vector<double> ax = matvec(A, x);
    double rr = 0, bb = 0;
    for (int i = 0; i < n; ++i) {
        double r = b[i] - ax[i];
        rr += r * r;
        bb += b[i] * b[i];
    }
    double rel_err = std::sqrt(en / tn);
    double rel_res = std::sqrt(rr) / std::sqrt(bb);
    std::printf("  n=%d nnz=%d  ||x-xtrue||/||xtrue||=%.3e  ||b-Ax||/||b||=%.3e\n", n,
                (int)A.vals.size(), rel_err, rel_res);

    check(rel_res < 1e-8, "AmgX residual ||b-Ax||/||b|| < 1e-8");
    check(rel_err < 1e-6, "AmgX recovers x_true to < 1e-6 (solves unstructured matrix)");
    return test_summary();
}
