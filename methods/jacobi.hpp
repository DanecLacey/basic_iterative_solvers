#pragma once

#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

void jacobi_fused_iteration(const MatrixCRS *A, const double *b, double *x_new,
                            const double *x_old) {
    double diag_elem;

#pragma omp parallel for schedule(static)
    for (int row_idx = 0; row_idx < A->n_rows; ++row_idx) {
        double sum = 0.0;
        int start_row = A->row_ptr[row_idx];
        int stop_row = A->row_ptr[row_idx + 1];

        for (int nz_idx = start_row; nz_idx < stop_row; ++nz_idx) {
            if (row_idx == A->col[nz_idx]) {
                diag_elem = A->val[nz_idx];
            } else {
                sum += A->val[nz_idx] * x_old[A->col[nz_idx]];
            }
        }
        x_new[row_idx] = (b[row_idx] - sum) / diag_elem;
    }
}

void normalize_x(double *x_new, const double *x_old, const double *D,
                 const double *b, const int n_rows) {
    double scaled_x_old;
    double adjusted_x;

#pragma omp parallel for schedule(static)
    for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
        scaled_x_old = D[row_idx] * x_old[row_idx];

        adjusted_x = x_new[row_idx] - scaled_x_old;

        x_new[row_idx] = (b[row_idx] - adjusted_x) / D[row_idx];
    }
}

// clang-format off
void jacobi_separate_iteration(Timers *timers, const MatrixCRS *A,
                               const double *D, const double *b, double *x_new,
                               const double *x_old, Interface *smax = nullptr) {

    // x_new <- A*x_old
    TIME(timers->spmv, spmv(A, x_old, x_new SMAX_ARGS(0, smax, "x_new <- A*x_old")));

    // x_new <- D^{-1}(b - (x_new - D))
    TIME(timers->normalize, normalize_x(x_new, x_old, D, b, A->n_rows))
}

class JacobiSolver : public Solver {
  public:
    // Solver-specific fields
    double *x_new = nullptr;
    double *x_old = nullptr;

    JacobiSolver(const Args *cli_args) : Solver(cli_args) {
        // Jacobi-specific initialization?
    }

    void allocate_structs(const int N) override {
        Solver::allocate_structs(N);
        x_new = new double[N];
        x_old = new double[N];
    }

    void init_structs(const int N) override {
        Solver::init_structs(N);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            x_new[i] = 0.0;
            x_old[i] = x_0[i];
        }
    }

    void init_residual() override {
        compute_residual(A.get(), x_old, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        // residual_norm = infty_vec_norm(residual, A->n_cols);
        residual_norm = euclidean_vec_norm(residual, A->n_cols);
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
        jacobi_separate_iteration(timers, A.get(), A_D, b, x_new, x_old SMAX_ARGS(smax));
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
        compute_residual(A.get(), x_new, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        // residual_norm = infty_vec_norm(residual, A->n_cols);
        residual_norm = euclidean_vec_norm(residual, A->n_cols);
        Solver::record_residual_norm();
    }
    // clang-format on

#ifdef USE_SMAX
    void register_structs() override {
        int N = A->n_cols;
        register_spmv(smax, "residual_spmv", A.get(), x_old, N, tmp, N);
        register_spmv(smax, "x_new <- A*x_old", A.get(), x_old, N, x_new, N);
    }
#endif

    ~JacobiSolver() override {
        delete[] x_new;
        delete[] x_old;
    }
};
