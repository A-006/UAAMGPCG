/**
 * @file amgx_backend.h
 * @brief Shared NVIDIA AMGX CSR solve engine (used by the 2D and 3D backends).
 * @author liutao
 * @date 2026-06-23
 *
 * Owns the AMGX handles (config/resources/matrix/vectors/solver) and solves a
 * single sparse system A x = b given in CSR. All AMGX types are hidden behind a
 * pimpl, so this header has no AMGX dependency and callers need no AMGX include.
 *
 * Constructing this when the project was built without AMGX support
 * (UAAMG_WITH_AMGX=OFF) throws a clear runtime_error.
 */
#pragma once
#include <memory>
#include <string>
#include <vector>

class AmgxBackend {
public:
    AmgxBackend();
    ~AmgxBackend();

    /**
     * @brief Solve A x = b on the GPU via AMGX.
     * @param n        Number of rows/unknowns.
     * @param row_ptr  CSR row offsets (size n+1), 0-based.
     * @param col_idx  CSR column indices (size nnz), 0-based, sorted per row.
     * @param vals     CSR values (size nnz).
     * @param b        Right-hand side (size n).
     * @param max_iter Maximum AMGX iterations.
     * @param tol      Relative residual tolerance (RELATIVE_INI, L2).
     * @return Solution x (size n).
     */
    std::vector<double> solve_csr(int n, const std::vector<int>& row_ptr,
                                  const std::vector<int>& col_idx, const std::vector<double>& vals,
                                  const std::vector<double>& b, int max_iter, double tol);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
