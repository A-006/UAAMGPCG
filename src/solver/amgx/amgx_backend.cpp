/**
 * @file amgx_backend.cpp
 * @brief AMGX CSR solve engine (see amgx_backend.h).
 * @author liutao
 * @date 2026-06-23
 */
#include "solver/amgx/amgx_backend.h"

#include <algorithm>
#include <stdexcept>

#ifdef HAVE_AMGX
#include "amgx_c.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

void check(AMGX_RC rc, const char* what) {
    if (rc != AMGX_RC_OK) {
        char msg[512];
        AMGX_get_error_string(rc, msg, sizeof(msg));
        throw std::runtime_error(std::string("AMGX error in ") + what + ": " + msg);
    }
}

// AMGX must be initialised once per process.
void amgx_global_init() {
    static std::once_flag once;
    std::call_once(once, [] { check(AMGX_initialize(), "AMGX_initialize"); });
}

// Default config: classical-AMG-preconditioned PCG — the combination verified to
// converge on the UAAMGPCG pressure systems. max_iters / tolerance are filled in
// per solve from the call arguments.
std::string default_config(int max_iter, double tol, bool verbose) {
    char buf[2048];
    std::snprintf(buf, sizeof(buf),
        "{"
        "\"config_version\": 2,"
        "\"solver\": {"
        "  \"solver\": \"PCG\","
        "  \"preconditioner\": {"
        "    \"solver\": \"AMG\","
        "    \"algorithm\": \"CLASSICAL\","
        "    \"max_iters\": 1,"
        "    \"cycle\": \"V\","
        "    \"presweeps\": 1,"
        "    \"postsweeps\": 1,"
        "    \"aggressive_levels\": 2,"
        "    \"interpolator\": \"D2\","
        "    \"max_levels\": 50,"
        "    \"smoother\": { \"solver\": \"BLOCK_JACOBI\" }"
        "  },"
        "  \"max_iters\": %d,"
        "  \"tolerance\": %.3e,"
        "  \"convergence\": \"RELATIVE_INI\","
        "  \"norm\": \"L2\","
        "  \"monitor_residual\": 1,"
        "  \"obtain_timings\": 0,"
        "  \"print_solve_stats\": %d"
        "}}",
        std::max(1, max_iter), std::max(tol, 1e-14), verbose ? 1 : 0);
    return buf;
}

} // namespace

struct AmgxBackend::Impl {
    AMGX_config_handle cfg     = nullptr;
    AMGX_resources_handle rsrc = nullptr;
    AMGX_matrix_handle A       = nullptr;
    AMGX_vector_handle x       = nullptr;
    AMGX_vector_handle b       = nullptr;
    AMGX_solver_handle solver  = nullptr;
    const AMGX_Mode mode       = AMGX_mode_dDDI;
    int cfg_max_iter           = -1;
    double cfg_tol             = -1;
    bool verbose               = false;
    bool from_file             = false;

    Impl() {
        amgx_global_init();
        verbose = std::getenv("UAAMG_AMGX_VERBOSE") != nullptr;
        if (const char* f = std::getenv("UAAMG_AMGX_CONFIG")) {
            check(AMGX_config_create_from_file(&cfg, f), "config_create_from_file");
            from_file = true;
        }
    }

    // (Re)create config + handles when the requested max_iter/tol change. With an
    // external config file the parameters come from the file (handles built once).
    void ensure_handles(int max_iter, double tol) {
        if (solver && (from_file || (max_iter == cfg_max_iter && tol == cfg_tol)))
            return;
        destroy_solve_handles();
        if (!from_file) {
            if (cfg)
                AMGX_config_destroy(cfg);
            std::string s = default_config(max_iter, tol, verbose);
            check(AMGX_config_create(&cfg, s.c_str()), "config_create");
            cfg_max_iter = max_iter;
            cfg_tol      = tol;
        }
        check(AMGX_resources_create_simple(&rsrc, cfg), "resources_create_simple");
        check(AMGX_matrix_create(&A, rsrc, mode), "matrix_create");
        check(AMGX_vector_create(&x, rsrc, mode), "vector_create x");
        check(AMGX_vector_create(&b, rsrc, mode), "vector_create b");
        check(AMGX_solver_create(&solver, rsrc, mode, cfg), "solver_create");
    }

    void destroy_solve_handles() {
        if (solver) { AMGX_solver_destroy(solver); solver = nullptr; }
        if (b) { AMGX_vector_destroy(b); b = nullptr; }
        if (x) { AMGX_vector_destroy(x); x = nullptr; }
        if (A) { AMGX_matrix_destroy(A); A = nullptr; }
        if (rsrc) { AMGX_resources_destroy(rsrc); rsrc = nullptr; }
    }

    ~Impl() {
        destroy_solve_handles();
        if (cfg)
            AMGX_config_destroy(cfg);
        // AMGX_finalize() intentionally not called: AMGX is a process-wide
        // singleton and other backend instances may still be alive.
    }
};

AmgxBackend::AmgxBackend() : impl_(std::make_unique<Impl>()) {}
AmgxBackend::~AmgxBackend() = default;

std::vector<double> AmgxBackend::solve_csr(int n, const std::vector<int>& row_ptr,
                                           const std::vector<int>& col_idx,
                                           const std::vector<double>& vals,
                                           const std::vector<double>& b, int max_iter, double tol) {
    std::vector<double> xh(n, 0.0);
    if (n == 0)
        return xh;

    impl_->ensure_handles(max_iter, tol);
    const int nnz = static_cast<int>(col_idx.size());
    check(AMGX_matrix_upload_all(impl_->A, n, nnz, 1, 1, row_ptr.data(), col_idx.data(),
                                 vals.data(), nullptr),
          "matrix_upload_all");
    check(AMGX_vector_upload(impl_->b, n, 1, b.data()), "vector_upload b");
    check(AMGX_vector_set_zero(impl_->x, n, 1), "vector_set_zero x");
    check(AMGX_solver_setup(impl_->solver, impl_->A), "solver_setup");
    check(AMGX_solver_solve(impl_->solver, impl_->b, impl_->x), "solver_solve");
    check(AMGX_vector_download(impl_->x, xh.data()), "vector_download");
    return xh;
}

#else // !HAVE_AMGX ─────────────────────────────────────────────────────────────

struct AmgxBackend::Impl {};

AmgxBackend::AmgxBackend() {
    throw std::runtime_error(
        "AMGX solver requested but the project was built without AMGX support. "
        "Reconfigure with -DUAAMG_WITH_AMGX=ON -DAMGX_ROOT=/path/to/AMGX (built).");
}
AmgxBackend::~AmgxBackend() = default;

std::vector<double> AmgxBackend::solve_csr(int, const std::vector<int>&, const std::vector<int>&,
                                           const std::vector<double>&, const std::vector<double>&,
                                           int, double) {
    return {};
}

#endif
