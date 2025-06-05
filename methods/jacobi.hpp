#pragma once

#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

void jacobi_fused_iteration(const MatrixCRS *crs_mat, const double *b,
                            double *x_new, const double *x_old) {
    double diag_elem;

#pragma omp parallel for
    for (int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx) {
        double sum = 0.0;
        int start_row = crs_mat->row_ptr[row_idx];
        int stop_row = crs_mat->row_ptr[row_idx + 1];

        for (int nz_idx = start_row; nz_idx < stop_row; ++nz_idx) {
            if (row_idx == crs_mat->col[nz_idx]) {
                diag_elem = crs_mat->val[nz_idx];
            } else {
                sum += crs_mat->val[nz_idx] * x_old[crs_mat->col[nz_idx]];
            }
        }
        x_new[row_idx] = (b[row_idx] - sum) / diag_elem;
    }
}

void normalize_x(double *x_new, const double *x_old, const double *D,
                 const double *b, const int n_rows) {
    double scaled_x_old;
    double adjusted_x;

#pragma omp parallel for
    for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
        scaled_x_old = D[row_idx] * x_old[row_idx];

        adjusted_x = x_new[row_idx] - scaled_x_old;

        x_new[row_idx] = (b[row_idx] - adjusted_x) / D[row_idx];
    }
}

// clang-format off
void jacobi_separate_iteration(Timers *timers, const MatrixCRS *crs_mat,
                               const double *D, const double *b, double *x_new,
                               const double *x_old, Interface *smax = nullptr) {

    // x_new <- A*x_old
    TIME(timers->spmv, spmv(crs_mat, x_old, x_new SMAX_ARGS(0, smax, "x_new <- A*x_old")));

    // x_new <- D^{-1}(b - (x_new - D))
    TIME(timers->normalize, normalize_x(x_new, x_old, D, b, crs_mat->n_rows))
}

class JacobiSolver : public Solver {
  public:
    // Solver-specific fields
    double *x_new = nullptr;
    double *x_old = nullptr;

    JacobiSolver(const Args *cli_args) : Solver(cli_args) {
        // Jacobi-specific initialization?
    }

    void allocate_structs() override {
        Solver::allocate_structs();
        x_new = new double[crs_mat->n_cols];
        x_old = new double[crs_mat->n_cols];
    }

    void init_structs() override {
        Solver::init_structs();
#pragma omp parallel for
        for (int i = 0; i < crs_mat->n_cols; ++i) {
            x_new[i] = 0.0;
            x_old[i] = x_0[i];
        }
    }

    void init_residual() override {
        compute_residual(crs_mat, x_old, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        residual_norm = infty_vec_norm(residual, crs_mat->n_cols);
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
        jacobi_separate_iteration(timers, crs_mat, D, b, x_new, x_old SMAX_ARGS(smax));
    }

    void exchange() override {
        std::swap(x_old, x_new);
#ifdef USE_SMAX
        this->smax->kernel("x_new <- A*x_old")->swap_operands();
#endif
    }

    void save_x_star() override {
        std::swap(x_old, x_star);
        Solver::save_x_star();
    }

    void record_residual_norm() override {
        compute_residual(crs_mat, x_new, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        residual_norm = infty_vec_norm(residual, crs_mat->n_cols);
        Solver::record_residual_norm();
    }
    // clang-format on

#ifdef USE_SMAX
    void register_structs() override {
        int N = crs_mat->n_cols;
        register_spmv(smax, "residual_spmv", crs_mat, x_old, N, tmp, N);
        register_spmv(smax, "x_new <- A*x_old", crs_mat, x_old, N, x_new, N);
    }
#endif

    ~JacobiSolver() override {
        delete[] x_new;
        delete[] x_old;
    }
};
