#pragma once

#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

// https://jiechenjiechen.github.io/pub/fbcgs.pdf
// https://doi.org/10.1137/0913035
void bicgstab_separate_iteration(
    Timers *timers, const PrecondType preconditioner, const MatrixCRS *A,
    const MatrixCRS *L, const MatrixCRS *U, double *A_D, double *A_D_inv,
    double *L_D, double *U_D, double *x_new, double *x_old, double *tmp,
    double *work, double *p_new, double *p_old, double *residual_new,
    double *residual_old, double *residual_0, double *v, double *h, double *s,
    double *s_tmp, double *t, double *t_tmp, double *y, double *z,
    double &rho_new, double rho_old, Interface *smax = nullptr) {

    int N = A->n_cols;

    IF_DEBUG_MODE_FINE(SanityChecker::print_bicgstab_vectors(
        A->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new, residual_old,
        residual_0, A_D, v, h, s, t, rho_new, rho_old, "before"))

    // y <- M^{-1}p_old
    TIME(timers->precond,
         apply_preconditioner(preconditioner, N, L, U, A_D, A_D_inv, L_D, U_D,
                              y, p_old, tmp,
                              work SMAX_ARGS(0, smax, "M^{-1} * p_old")))

    // v <- A*y
    TIME(timers->spmv, spmv(A, y, v SMAX_ARGS(0, smax, "v <- A*y")))

    // alpha <- rho_old / (r_0, v)
    double alpha;
    TIME(timers->dot, alpha = rho_old / dot(residual_0, v, N))

    IF_DEBUG_MODE_FINE(printf("alpha = %f\n", alpha))

    // s <- r_old - alpha*v
    TIME(timers->sum, subtract_vectors(s, residual_old, v, N, alpha))

    // s_tmp <- M^{-1}s
    TIME(timers->precond,
         apply_preconditioner(preconditioner, N, L, U, A_D, A_D_inv, L_D, U_D,
                              s_tmp, s, tmp,
                              work SMAX_ARGS(0, smax, "M^{-1} * s")))

    // z <- A*s_tmp
    TIME(timers->spmv, spmv(A, s_tmp, z SMAX_ARGS(0, smax, "z <- A*s_tmp")))

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
        A->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new, residual_old,
        residual_0, A_D, v, h, s, t, rho_new, rho_old, "after"))
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

    void allocate_structs(const int N) override {
        Solver::allocate_structs(N);
        x_new = new double[N];
        x_old = new double[N];
        p_new = new double[N];
        p_old = new double[N];
        residual_new = new double[N];
        residual_old = new double[N];
        v = new double[N];
        h = new double[N];
        s = new double[N];
        s_tmp = new double[N];
        t = new double[N];
        t_tmp = new double[N];
        y = new double[N];
        z = new double[N];
    }

    void init_structs(const int N) override {
        Solver::init_structs(N);
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
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
        compute_residual(A.get(), x_old, b, residual,
                         tmp SMAX_ARGS(smax, "residual_spmv"));

        copy_vector(residual_old, residual, A->n_cols);
        residual_norm = euclidean_vec_norm(residual, A->n_cols);

        // Precondition the initial residual
        IF_DEBUG_MODE_FINE(SanityChecker::print_vector(
            residual, A->n_cols, "residual before preconditioning"));
        apply_preconditioner(preconditioner, A->n_cols, L_strict.get(),
                             U_strict.get(), A_D, A_D_inv, L_D, U_D, residual,
                             residual, tmp,
                             work SMAX_ARGS(0, smax, "init M^{-1} * residual"));
        IF_DEBUG_MODE_FINE(SanityChecker::print_vector(
            residual, A->n_cols, "residual after preconditioning"));

        // Make copies of initial residual for solver
        copy_vector(p_old, residual, A->n_cols);

        rho_old = dot(residual_old, residual, A->n_cols);
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
        bicgstab_separate_iteration(
            timers, preconditioner, A.get(), L_strict.get(), U_strict.get(),
            A_D, A_D_inv, L_D, U_D, x_new, x_old, tmp, work, p_new, p_old,
            residual_new, residual_old, residual_0, v, h, s, s_tmp, t, t_tmp, y,
            z, rho_new, rho_old SMAX_ARGS(smax));
        std::swap(residual, residual_new);
    }

    // clang-format off
    void exchange() override {
        std::swap(p_old, p_new);
        std::swap(residual_old, residual); // <- swapped r and r_new earlier
        std::swap(x_old, x_new);
        std::swap(rho_old, rho_new);
#ifdef USE_SMAX
        if (preconditioner == PrecondType::GaussSeidel ||
            preconditioner == PrecondType::BackwardsGaussSeidel) {
            auto *sptrsv1 = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * p_old"));
            sptrsv1->args->x->val = static_cast<void *>(y);
            sptrsv1->args->y->val = static_cast<void *>(p_old);
            auto *sptrsv2 = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * s"));
            sptrsv2->args->x->val = static_cast<void *>(s_tmp);
            sptrsv2->args->y->val = static_cast<void *>(s);
        } else if (
            preconditioner == PrecondType::SymmetricGaussSeidel ||
            preconditioner == PrecondType::ILU0 ||  
            preconditioner == PrecondType::ILUT
        ) {
            auto *sptrsv1_lower = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * p_old_lower"));
            sptrsv1_lower->args->x->val = static_cast<void *>(tmp);
            sptrsv1_lower->args->y->val = static_cast<void *>(p_old);
            auto *sptrsv1_upper = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * p_old_upper"));
            sptrsv1_upper->args->x->val = static_cast<void *>(y);
            sptrsv1_upper->args->y->val = static_cast<void *>(tmp);
            auto *sptrsv2_lower = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * s_lower"));
            sptrsv2_lower->args->x->val = static_cast<void *>(tmp);
            sptrsv2_lower->args->y->val = static_cast<void *>(s);
            auto *sptrsv2_upper = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * s_upper"));
            sptrsv2_upper->args->x->val = static_cast<void *>(s_tmp);
            sptrsv2_upper->args->y->val = static_cast<void *>(tmp);
        }
#endif
    }

    void save_x_star() override {
        std::swap(x_old, x_star);
        Solver::save_x_star();
    }

    void record_residual_norm() override {
        residual_norm = euclidean_vec_norm(residual, A->n_cols);
        Solver::record_residual_norm();
    }

#ifdef USE_SMAX
    void register_structs() override {
        int N = A->n_cols;
        register_spmv(smax, "residual_spmv", A.get(), x_old, N, tmp, N);
        register_spmv(smax, "v <- A*y", A.get(), y, N, v, N);
        register_spmv(smax, "z <- A*s_tmp", A.get(), s_tmp, N, z, N);
        if (preconditioner == PrecondType::GaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual", L.get(), residual, N, residual_old, N);
            register_sptrsv(smax, "M^{-1} * p_old", L.get(), y, N, p_old, N);
            register_sptrsv(smax, "M^{-1} * s", L.get(), s_tmp, N, s, N);
        } else if (preconditioner == PrecondType::BackwardsGaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual", U.get(), residual, N, residual_old, N, true);
            register_sptrsv(smax, "M^{-1} * p_old", U.get(), y, N, p_old, N, true);
            register_sptrsv(smax, "M^{-1} * s", U.get(), s_tmp, N, s, N, true);
        } else if (
            preconditioner == PrecondType::SymmetricGaussSeidel ||
            preconditioner == PrecondType::ILU0 ||  
            preconditioner == PrecondType::ILUT
        ) {
            register_sptrsv(smax, "init M^{-1} * residual_lower", L.get(), tmp, N, residual_old, N);
            register_sptrsv(smax, "init M^{-1} * residual_upper", U.get(), residual, N, tmp, N, true);
            register_sptrsv(smax, "M^{-1} * p_old_lower", L.get(), tmp, N, p_old, N);
            register_sptrsv(smax, "M^{-1} * p_old_upper", U.get(), y, N, tmp, N, true);
            register_sptrsv(smax, "M^{-1} * s_lower", L.get(), tmp, N, s, N);
            register_sptrsv(smax, "M^{-1} * s_upper", U.get(), s_tmp, N, tmp, N, true);
        } else if (preconditioner == PrecondType::TwoStageGS) {
            register_spmv(smax, "init M^{-1} * residual", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * p_old", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * s", L_strict.get(), work, N, tmp, N);
        } else if (preconditioner == PrecondType::SymmetricTwoStageGS) {
            register_spmv(smax, "init M^{-1} * residual_lower", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "init M^{-1} * residual_upper", U_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * p_old_lower", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * p_old_upper", U_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * s_lower", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * s_upper", U_strict.get(), work, N, tmp, N);
        }
    }
#endif
    // clang-format on

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
