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
    MatrixCRS *crs_mat = nullptr;
    MatrixCRS *crs_mat_L = nullptr;
    MatrixCRS *crs_mat_L_strict = nullptr;
    MatrixCRS *crs_mat_U = nullptr;
    MatrixCRS *crs_mat_U_strict = nullptr;

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
    double *residual = nullptr;
    double *residual_0 = nullptr;
    double *D = nullptr;

    // Bookkeeping
    double *collected_residual_norms = nullptr;
    double *time_per_iteration = nullptr;

    // Useful flags
    bool convergence_flag = false;
    bool gmres_restarted = false;

    Solver(const Args *cli_args)
        : method(cli_args->method), preconditioner(cli_args->preconditioner) {
        collected_residual_norms = new double[max_iters];
        time_per_iteration = new double[max_iters];

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

    // Partially overridden
    virtual void allocate_structs() {
        x_star = new double[crs_mat->n_cols];
        x_0 = new double[crs_mat->n_cols];
        b = new double[crs_mat->n_cols];
        tmp = new double[crs_mat->n_cols];
        residual = new double[crs_mat->n_cols];
        residual_0 = new double[crs_mat->n_cols];
        D = new double[crs_mat->n_cols];

        if (!gmres_restarted) {
            // NOTE: We don't want to overwrite these when restarting GMRES
#pragma omp parallel for
            for (int i = 0; i < crs_mat->n_cols; ++i) {
                x_star[i] = 0.0;
                x_0[i] = INIT_X_VAL;
                b[i] = B_VAL;
                D[i] = 0.0;
            }
        }
    }

    virtual void check_restart() {
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
        delete[] residual;
        delete[] residual_0;
        delete[] D;
        delete[] collected_residual_norms;
        delete[] time_per_iteration;
    }

    virtual void init_structs() {
#pragma omp parallel for
        for (int i = 0; i < crs_mat->n_cols; ++i) {
            tmp[i] = 0.0;
            residual[i] = 0.0;
            residual_0[i] = 0.0;
        }
    }

    // clang-format off
    virtual void init_residual() {
        copy_vector(residual_0, residual, crs_mat->n_cols);
        collected_residual_norms[collected_residual_norms_count] = residual_norm;
    }

    virtual void save_x_star() {
        compute_residual(crs_mat, x_star, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));
        residual_norm = infty_vec_norm(residual, crs_mat->n_cols);
        collected_residual_norms[collected_residual_norms_count] = residual_norm;
    }

    virtual void record_residual_norm() {
        collected_residual_norms[collected_residual_norms_count + 1] = residual_norm;
    };

    // Base class methods, not overridden //
    void sample_residual(Stopwatch *per_iteration_time) {
        if (iter_count % residual_check_len == 0) {
            record_residual_norm();
            time_per_iteration[collected_residual_norms_count] = per_iteration_time->check();
            ++collected_residual_norms_count;
        }
    }

    void init_stopping_criteria() {
        stopping_criteria = tolerance * residual_norm;
    }

    bool check_stopping_criteria() {
        bool norm_convergence = residual_norm < stopping_criteria;
        bool over_max_iters = iter_count >= max_iters;
        bool divergence = residual_norm > DBL_MAX;
        IF_DEBUG_MODE_FINE(
			if (norm_convergence)
            	printf("norm convergence met: %f < %f\n", residual_norm, stopping_criteria)
			if (over_max_iters)
				printf("over max iters: %i >= %i\n", iter_count, max_iters)
			if (divergence)
				printf("divergence\n")
		)
        return norm_convergence || over_max_iters || divergence;
    }
    // clang-format on
};
