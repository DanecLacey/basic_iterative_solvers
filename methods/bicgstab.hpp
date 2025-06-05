#pragma once

#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

// https://jiechenjiechen.github.io/pub/fbcgs.pdf
void bicgstab_separate_iteration(
    Timers *timers, const PrecondType preconditioner, const MatrixCRS *crs_mat,
    const MatrixCRS *crs_mat_L, const MatrixCRS *crs_mat_U, double *D,
    double *x_new, double *x_old, double *tmp, double *p_new, double *p_old,
    double *residual_new, double *residual_old, double *residual_0, double *v,
    double *h, double *s, double *s_tmp, double *t, double *t_tmp, double *y,
    double *z, double &rho_new, double rho_old, Interface *smax = nullptr) {

    int N = crs_mat->n_cols;

    IF_DEBUG_MODE_FINE(SanityChecker::print_bicgstab_vectors(
        crs_mat->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new,
        residual_old, residual_0, v, h, s, t, rho_new, rho_old, "before"))

    // y <- M^{-1}p_old
    TIME(timers->precond, apply_preconditioner(preconditioner, crs_mat_L,
                                               crs_mat_U, D, y, p_old, tmp))

    // v <- A*y
    TIME(timers->spmv, spmv(crs_mat, y, v SMAX_ARGS(0, smax, "v <- A*y")))

    // alpha <- rho_old / (r_0, v)
    double alpha;
    TIME(timers->dot, alpha = rho_old / dot(residual_0, v, N))

    IF_DEBUG_MODE_FINE(printf("alpha = %f\n", alpha))

    // s <- r_old - alpha*v
    TIME(timers->sum, subtract_vectors(s, residual_old, v, N, alpha))

    // s_tmp <- M^{-1}s
    TIME(timers->precond, apply_preconditioner(preconditioner, crs_mat_L,
                                               crs_mat_U, D, s_tmp, s, tmp))

    // z <- A*s_tmp
    TIME(timers->spmv,
         spmv(crs_mat, s_tmp, z SMAX_ARGS(0, smax, "z <- A*s_tmp")))

    double omega;
    TIME(timers->dot, omega = dot(z, s, N) / dot(z, z, N))

    // h <- x_old + alpha*y
    TIME(timers->sum, sum_vectors(h, x_old, y, N, alpha))

    // TODO: quit if s is small enough

    IF_DEBUG_MODE_FINE(printf("omega = %f\n", omega))

    // x_new <- h + omega*s_tmp
    TIME(timers->sum, sum_vectors(x_new, h, s_tmp, N, omega))

    // r_new <- s - omega*z
    TIME(timers->sum, subtract_vectors(residual_new, s, z, N, omega))

    // TODO: quit if r_new is small enough

    TIME(timers->dot, rho_new = dot(residual_0, residual_new, N))

    double beta = (rho_new / rho_old) * (alpha / omega);

    IF_DEBUG_MODE_FINE(printf("beta = %f\n", beta))

    // tmp <- p_old - omega*v
    TIME(timers->sum, subtract_vectors(tmp, p_old, v, N, omega))

    // p_new <- r_new + beta*tmp
    TIME(timers->sum, sum_vectors(p_new, residual_new, tmp, N, beta))

    IF_DEBUG_MODE_FINE(SanityChecker::print_bicgstab_vectors(
        crs_mat->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new,
        residual_old, residual_0, v, h, s, t, rho_new, rho_old, "after"))
}

class BiCGSTABSolver : public Solver {

  public:
    // Solver-specific fields
    double *x_new = nullptr;
    double *x_old = nullptr;
    double *p_old = nullptr;
    double *p_new = nullptr;
    double *v = nullptr;
    double *h = nullptr;
    double *s = nullptr;
    double *s_tmp = nullptr;
    double *t = nullptr;
    double *t_tmp = nullptr;
    double rho_old = 0.0;
    double rho_new = 0.0;
    double *z = nullptr;
    double *y = nullptr;
    double *residual_old = nullptr;
    double *residual_new = nullptr;

    BiCGSTABSolver(const Args *cli_args) : Solver(cli_args) {
        // BiCGSTAB-specific initialization?
    }

    void allocate_structs() override {
        Solver::allocate_structs();
        x_new = new double[crs_mat->n_cols];
        x_old = new double[crs_mat->n_cols];
        p_new = new double[crs_mat->n_cols];
        p_old = new double[crs_mat->n_cols];
        residual_new = new double[crs_mat->n_cols];
        residual_old = new double[crs_mat->n_cols];
        v = new double[crs_mat->n_cols];
        h = new double[crs_mat->n_cols];
        s = new double[crs_mat->n_cols];
        s_tmp = new double[crs_mat->n_cols];
        t = new double[crs_mat->n_cols];
        t_tmp = new double[crs_mat->n_cols];
        y = new double[crs_mat->n_cols];
        z = new double[crs_mat->n_cols];
    }

    void init_structs() override {
        Solver::init_structs();
#pragma omp parallel for
        for (int i = 0; i < crs_mat->n_cols; ++i) {
            x_new[i] = 0.0;
            x_old[i] = x_0[i];
            p_new[i] = 0.0;
            p_old[i] = 0.0;
            residual_new[i] = 0.0;
            residual_old[i] = 0.0;
            v[i] = 0.0;
            h[i] = 0.0;
            s[i] = 0.0;
            s_tmp[i] = 0.0;
            t[i] = 0.0;
            t_tmp[i] = 0.0;
        }
    }

    void init_residual() override {
        compute_residual(crs_mat, x_old, b, residual,
                         tmp SMAX_ARGS(smax, "residual_spmv"));

        // Make copies of initial residual for solver
        copy_vector(p_old, residual, crs_mat->n_cols);
        copy_vector(residual_old, residual, crs_mat->n_cols);
        residual_norm = infty_vec_norm(residual, crs_mat->n_cols);
        rho_old = dot(residual_old, residual, crs_mat->n_cols);
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
        bicgstab_separate_iteration(
            timers, preconditioner, crs_mat, crs_mat_L_strict, crs_mat_U_strict,
            D, x_new, x_old, tmp, p_new, p_old, residual_new, residual_old,
            residual_0, v, h, s, s_tmp, t, t_tmp, y, z, rho_new,
            rho_old SMAX_ARGS(smax));
        std::swap(residual, residual_new);
    }

    void exchange() override {
        std::swap(p_old, p_new);
        std::swap(residual_old, residual); // <- swapped r and r_new earlier
        std::swap(x_old, x_new);
        std::swap(rho_old, rho_new);
    }

    void save_x_star() override {
        std::swap(x_old, x_star);
        Solver::save_x_star();
    }

    void record_residual_norm() override {
        residual_norm = infty_vec_norm(residual, crs_mat->n_cols);
        Solver::record_residual_norm();
    }

#ifdef USE_SMAX
    void register_structs() override {
        int N = crs_mat->n_cols;
        register_spmv(smax, "residual_spmv", crs_mat, x_old, N, tmp, N);
        register_spmv(smax, "v <- A*y", crs_mat, y, N, v, N);
        register_spmv(smax, "z <- A*s_tmp", crs_mat, s_tmp, N, z, N);
    }
#endif

    ~BiCGSTABSolver() override {
        delete[] x_new;
        delete[] x_old;
        delete[] p_new;
        delete[] p_old;
        delete[] residual_new;
        delete[] residual_old;
        delete[] v;
        delete[] h;
        delete[] s;
        delete[] s_tmp;
        delete[] t;
        delete[] t_tmp;
        delete[] y;
        delete[] z;
    }
};
