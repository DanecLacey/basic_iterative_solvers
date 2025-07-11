#pragma once

#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

void gs_fused_iteration(const MatrixCRS *crs_mat, const double *b, double *x) {
    double diag_elem = 1.0;

    for (int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx) {
        double sum = 0.0;
        int start_row = crs_mat->row_ptr[row_idx];
        int stop_row = crs_mat->row_ptr[row_idx + 1];

        for (int nz_idx = start_row; nz_idx < stop_row; ++nz_idx) {
            if (row_idx == crs_mat->col[nz_idx]) {
                diag_elem = crs_mat->val[nz_idx];
            } else {
                sum += crs_mat->val[nz_idx] * x[crs_mat->col[nz_idx]];
            }
        }
        x[row_idx] = (b[row_idx] - sum) / diag_elem;
    }
}

// clang-format off
void gs_separate_iteration(Timers *timers, const MatrixCRS *crs_mat_U,
                           const MatrixCRS *crs_mat_L, double *tmp,
                           const double *D, const double *b, double *x,
                           Interface *smax = nullptr) {
    // tmp <- U*x
    TIME(timers->spmv, spmv(crs_mat_U, x, tmp SMAX_ARGS(0, smax, "tmp <- U*x")))

    // tmp <- b - tmp
    TIME(timers->sum, subtract_vectors(tmp, b, tmp, crs_mat_U->n_rows))

    // x <- (D+L)^{-1}(tmp)
    TIME(timers->sptrsv, sptrsv(crs_mat_L, x, D, tmp SMAX_ARGS(0, smax, "solve x <- (D+L)^{-1}(b-Ux)")))
}

void bgs_separate_iteration(Timers *timers, const MatrixCRS *crs_mat_U,
                            const MatrixCRS *crs_mat_L, double *tmp,
                            const double *D, const double *b, double *x,
                            Interface *smax = nullptr) {
    // tmp <- L*x
    TIME(timers->spmv, spmv(crs_mat_L, x, tmp SMAX_ARGS(0, smax, "tmp <- L*x")))

    // tmp <- b - tmp
    TIME(timers->sum, subtract_vectors(tmp, b, tmp, crs_mat_L->n_rows))

    // x <- (D+U)^{-1}(tmp)
    TIME(timers->sptrsv, bsptrsv(crs_mat_U, x, D, tmp SMAX_ARGS(0, smax, "solve x <- (U+L)^{-1}(b-Ux)")))
}

class GaussSeidelSolver : public Solver {
  public:
    // Solver-specific fields
    double *x = nullptr;

    GaussSeidelSolver(const Args *cli_args) : Solver(cli_args) {
        // GaussSeidel-specific initialization?
    }

    void allocate_structs(const int N) override {
        Solver::allocate_structs(N);
        x = new double[N];
    }

    void init_structs(const int N) override {
        Solver::init_structs(N);
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            x[i] = x_0[i];
        }
    }

    void init_residual() override {
        compute_residual(crs_mat.get(), x, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        // residual_norm = infty_vec_norm(residual, crs_mat->n_cols);
        residual_norm = euclidean_vec_norm(residual, crs_mat->n_cols);
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
        gs_separate_iteration(
            timers, crs_mat_U_strict.get(), crs_mat_L_strict.get(), tmp,
            D, b, x SMAX_ARGS(smax)
        );
    }

    void exchange() override {
        // Nothing to swap for Gauss-Seidel
    }

    void save_x_star() override {
        std::swap(x, x_star);
        Solver::save_x_star();
    }

    void record_residual_norm() override {
        compute_residual(crs_mat.get(), x, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        // residual_norm = infty_vec_norm(residual, crs_mat->n_cols);
        residual_norm = euclidean_vec_norm(residual, crs_mat->n_cols);
        Solver::record_residual_norm();
    }

#ifdef USE_SMAX
    void register_structs() override {
        int N = crs_mat->n_cols;
        register_spmv(smax, "residual_spmv", crs_mat.get(), x, N, tmp, N);
        register_spmv(smax, "tmp <- U*x", crs_mat_U_strict.get(), x, N, tmp, N);
        register_sptrsv(smax, "solve x <- (D+L)^{-1}(b-Ux)", crs_mat_L.get(), x, N, tmp, N);
    }
#endif

    ~GaussSeidelSolver() override { delete[] x; }
};

class SymmetricGaussSeidelSolver : public GaussSeidelSolver {

  public:
    SymmetricGaussSeidelSolver(const Args *cli_args)
        : GaussSeidelSolver(cli_args) {
        // SymmetricGaussSeidel-specific initialization?
    }

    void iterate(Timers *timers) override {
        gs_separate_iteration(timers, crs_mat_U_strict.get(), crs_mat_L_strict.get(), tmp, D, b, x SMAX_ARGS(smax));
        bgs_separate_iteration(timers, crs_mat_U_strict.get(), crs_mat_L_strict.get(), tmp, D, b, x SMAX_ARGS(smax));
    }

#ifdef USE_SMAX
    void register_structs() override {
        int N = crs_mat->n_cols;
        register_spmv(smax, "residual_spmv", crs_mat.get(), x, N, tmp, N);
        register_spmv(smax, "tmp <- U*x", crs_mat_U_strict.get(), x, N, tmp, N);
        register_sptrsv(smax, "solve x <- (D+L)^{-1}(b-Ux)", crs_mat_L.get(), x, N, tmp, N);
        register_spmv(smax, "tmp <- L*x", crs_mat_L_strict.get(), x, N, tmp, N);
        register_sptrsv(smax, "solve x <- (U+L)^{-1}(b-Ux)", crs_mat_U.get(), x, N, tmp, N, true);
    }
#endif
};
// clang-format on
