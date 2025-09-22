#pragma once

#include "common.hpp"
#include "kernels.hpp"
#include "sparse_matrix.hpp"

#include <float.h>

class Solver {
  public:
    SolverType method;
    PrecondType preconditioner = PrecondType::None;
#ifdef USE_SMAX
    SMAX::Interface *smax = nullptr;
#endif

    // Common structs
    std::unique_ptr<MatrixCRS> A;
    std::unique_ptr<MatrixCRS> L;
    std::unique_ptr<MatrixCRS> L_strict;
    std::unique_ptr<MatrixCRS> U;
    std::unique_ptr<MatrixCRS> U_strict;

    // Common parameters
    double stopping_criteria = 0.0;
    int iter_count = 0;
    int collected_residual_norms_count = 0;
    double residual_norm = DBL_MAX;
    int max_iters = MAX_ITERS;
    double tolerance = TOL;
    int residual_check_len = RES_CHECK_LEN;
    int gmres_restart_len = GMRES_RESTART_LEN;
    int gmres_restart_count = 0;

    // Common vectors
    double *x_star = nullptr;
    double *x_0 = nullptr;
    double *b = nullptr;
    double *tmp = nullptr;
    double *work = nullptr;
    double *residual = nullptr;
    double *residual_0 = nullptr;
    double *A_D = nullptr; // Diagonal of A
    double *A_D_inv = nullptr;
    double *L_D = nullptr; // Diagonal of L
    double *U_D = nullptr; // Diagonal of U

    // Bookkeeping
    double *collected_residual_norms = nullptr;
    double *time_per_iteration = nullptr;

    // Useful flags
    bool convergence_flag = false;
    bool gmres_restarted = false;

    Solver(const Args *cli_args)
        : method(cli_args->method), preconditioner(cli_args->preconditioner) {
        collected_residual_norms = new double[max_iters * 2];
        time_per_iteration = new double[max_iters * 2];

        for (int i = 0; i < max_iters; ++i) {
            collected_residual_norms[i] = 0.0;
            time_per_iteration[i] = 0.0;
        }
    }

    // Completely overridden //
    virtual void iterate(Timers *) = 0;
    virtual void exchange() = 0;
#ifdef USE_SMAX
    virtual void register_structs() = 0;
#endif

    // Partially overridden //
    virtual void allocate_structs(const int N) {
        x_star = new double[N];
        x_0 = new double[N];
        b = new double[N];
        tmp = new double[N];
        work = new double[N];
        residual = new double[N];
        residual_0 = new double[N];
        A_D = new double[N];
        A_D_inv = new double[N];
        L_D = new double[N];
        U_D = new double[N];

        if (!gmres_restarted) {
            // NOTE: We don't want to overwrite these when restarting GMRES
#pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                x_star[i] = 0.0;
                x_0[i] = INIT_X_VAL;
                b[i] = B_VAL;
                A_D[i] = 1.0; // div safe default
                A_D_inv[i] = 0.0;
                L_D[i] = 1.0; // div safe default
                U_D[i] = 1.0; // div safe default
            }
        }
    }

    virtual void init_structs(const int N) {
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            tmp[i] = 0.0;
            work[i] = 0.0;
            residual[i] = 0.0;
            residual_0[i] = 0.0;
        }
    }

    virtual void check_restart(Timers *) {
        // Do nothing by default
    };

    virtual void get_explicit_x() {
        // Do nothing by default
    };

    virtual ~Solver() {
        delete[] x_star;
        delete[] x_0;
        delete[] b;
        delete[] tmp;
        delete[] work;
        delete[] residual;
        delete[] residual_0;
        delete[] A_D;
        delete[] A_D_inv;
        delete[] L_D;
        delete[] U_D;
        delete[] collected_residual_norms;
        delete[] time_per_iteration;
    }

    // clang-format off
    virtual void init_residual() {
        copy_vector(residual_0, residual, A->n_cols);
        collected_residual_norms[collected_residual_norms_count++] = residual_norm;
    }

    virtual void save_x_star() {
        IF_DEBUG_MODE_FINE(SanityChecker::print_vector(x_star, A->n_rows, "x_star"));
        compute_residual(A.get(), x_star, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        // residual_norm = infty_vec_norm(residual, A->n_cols);
        residual_norm = euclidean_vec_norm(residual, A->n_cols);
        collected_residual_norms[collected_residual_norms_count + 1] = residual_norm;
    }

    virtual void record_residual_norm() {
        collected_residual_norms[collected_residual_norms_count++] = residual_norm;
    };

    // Base class methods, not overridden //
    void sample_residual(Stopwatch *per_iteration_time) {
        if (iter_count % residual_check_len == 0) {
            record_residual_norm();
            time_per_iteration[collected_residual_norms_count] = per_iteration_time->check();
        }
    }

    void init_stopping_criteria() {
        stopping_criteria = tolerance * residual_norm;
    }

    bool check_stopping_criteria() {
        bool norm_convergence = std::abs(residual_norm) < stopping_criteria;
        // We count GMRES restarts as an iteration
        bool over_max_iters = iter_count >= (max_iters - gmres_restart_count);
        bool divergence = std::abs(residual_norm) > DBL_MAX;
        IF_DEBUG_MODE_FINE(
			if (norm_convergence)
            	printf("norm convergence met: %f < %f\n", std::abs(residual_norm), stopping_criteria);
			if (over_max_iters)
				printf("over max iters: %i >= %i\n", iter_count, max_iters);
			if (divergence)
				printf("divergence\n");
		)
        return norm_convergence || over_max_iters || divergence;
    }
    // clang-format on
};
