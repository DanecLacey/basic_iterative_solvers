#pragma once

#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

// clang-format off
void cg_separate_iteration(Timers *timers, const PrecondType preconditioner,
                           const MatrixCRS *A, const MatrixCRS *L,
                           const MatrixCRS *U, double *D, double *D_inv, double *x_new,
                           double *x_old, double *tmp, double *work, double *p_new,
                           double *p_old, double *r_new, double *r_old,
                           double *z_new, double *z_old,
                           Interface *smax = nullptr) {

    // pre-compute tmp <- A*p_old
    TIME(timers->spmv, spmv(A, p_old, tmp SMAX_ARGS(0, smax, "tmp <- A*p_old")))

    double tmp_dot;
    TIME(timers->dot, tmp_dot = dot(r_old, z_old, A->n_cols))

    // alpha <- (r_old, z_old) / (Ap_old, p_old)
    double alpha;
    TIME(timers->dot, alpha = tmp_dot / dot(tmp, p_old, A->n_cols))

    IF_DEBUG_MODE_FINE(printf("alpha = %f\n", alpha))

    // x_new <- x_old + alpha * p_old
    TIME(timers->sum, sum_vectors(x_new, x_old, p_old, A->n_cols, alpha))

    // r_new <- r_old - alpha * Ap_old
    TIME(timers->sum, subtract_vectors(r_new, r_old, tmp, A->n_cols, alpha))

    // z_new <- M^{-1}r_new
    IF_DEBUG_MODE_FINE(SanityChecker::print_vector(z_new, A->n_cols, "z_new before preconditioning"));
    IF_DEBUG_MODE_FINE(SanityChecker::print_vector(r_new, A->n_cols, "r_new before preconditioning"));

    TIME(timers->precond,
     apply_preconditioner(
        preconditioner, A->n_cols, L, U, D, D_inv, z_new,
        r_new, tmp, work SMAX_ARGS(0, smax, "M^{-1} * residual"))
    )
    IF_DEBUG_MODE_FINE(SanityChecker::print_vector(z_new, A->n_cols, "z_new after preconditioning"));
    IF_DEBUG_MODE_FINE(SanityChecker::print_vector(r_new, A->n_cols, "r_new after preconditioning"));

    // beta <- (r_new, z_new) / (r_old, z_old)
    double beta;
    TIME(timers->dot, beta = dot(r_new, z_new, A->n_cols) / tmp_dot)

    IF_DEBUG_MODE_FINE(printf("beta = %f\n", beta))

    // p_new <- z_new + beta * p_old
    TIME(timers->sum, sum_vectors(p_new, z_new, p_old, A->n_cols, beta))
}

class ConjugateGradientSolver : public Solver {

  public:
    // Solver-specific fields
    double *x_new = nullptr;
    double *x_old = nullptr;
    double *p_old = nullptr;
    double *p_new = nullptr;
    double *z_old = nullptr;
    double *z_new = nullptr;
    double *residual_old = nullptr;
    double *residual_new = nullptr;

    ConjugateGradientSolver(const Args *cli_args) : Solver(cli_args) {
        // ConjugateGradient-specific initialization?
    }

    void allocate_structs(const int N) override {
        Solver::allocate_structs(N);
        x_new = new double[N];
        x_old = new double[N];
        p_new = new double[N];
        p_old = new double[N];
        residual_new = new double[N];
        residual_old = new double[N];
        z_new = new double[N];
        z_old = new double[N];
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
            z_new[i] = 0.0;
            z_old[i] = 0.0;
        }
    }

     void init_residual() override {
        compute_residual(A.get(), x_old, b, residual, tmp SMAX_ARGS(smax, "residual_spmv"));

        // Precondition the initial residual
        IF_DEBUG_MODE_FINE(SanityChecker::print_vector(residual, A->n_cols, "residual before preconditioning"));
        
        apply_preconditioner(
            preconditioner, A->n_cols, L_strict.get(), U_strict.get(), D, D_inv, z_old, 
            residual, tmp, work SMAX_ARGS(0, smax, "init M^{-1} * residual")
        );
        IF_DEBUG_MODE_FINE(SanityChecker::print_vector(z_old, A->n_cols, "residual after preconditioning"));

        // Make copies of initial residual for solver
        copy_vector(p_old, z_old, A->n_cols);
        copy_vector(residual_old, residual, A->n_cols);
        // residual_norm = infty_vec_norm(residual, A->n_cols);
        residual_norm = euclidean_vec_norm(residual, A->n_cols);
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
        cg_separate_iteration(timers, preconditioner, A.get(), L_strict.get(),
                              U_strict.get(), D, D_inv, x_new, x_old, tmp, work, p_new,
                              p_old, residual_new, residual_old, z_new,
                              z_old SMAX_ARGS(smax));
    }

    void exchange() override {
        std::swap(p_old, p_new);
        std::swap(z_old, z_new);
        std::swap(residual_old, residual_new);
        std::swap(x_old, x_new);
#ifdef USE_SMAX
        auto *spmv = dynamic_cast<SMAX::KERNELS::SpMVKernel *>(smax->kernel("tmp <- A*p_old"));
        spmv->args->x->val = static_cast<void *>(p_old);
        if (preconditioner == PrecondType::GaussSeidel || preconditioner == PrecondType::BackwardsGaussSeidel) {
            auto *sptrsv = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * residual"));
            sptrsv->args->x->val = static_cast<void *>(z_new);
            sptrsv->args->y->val = static_cast<void *>(residual_new);
        } else if (preconditioner == PrecondType::SymmetricGaussSeidel) {
            auto *lower_sptrsv = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * residual_lower"));
            lower_sptrsv->args->x->val = static_cast<void *>(tmp);
            lower_sptrsv->args->y->val = static_cast<void *>(residual_new);
            auto *upper_sptrsv = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * residual_upper"));
            upper_sptrsv->args->x->val = static_cast<void *>(z_new);
            upper_sptrsv->args->y->val = static_cast<void *>(tmp);
        }
#endif
    }

    void save_x_star() override {
        std::swap(x_old, x_star);
        Solver::save_x_star();
    }

    void record_residual_norm() override {
        // residual_norm = infty_vec_norm(residual_new, A->n_cols);
        residual_norm = euclidean_vec_norm(residual_new, A->n_cols);
        Solver::record_residual_norm();
    }

#ifdef USE_SMAX
    void register_structs() override {
        int N = A->n_cols;
        register_spmv(smax, "residual_spmv", A.get(), x_old, N, tmp, N);
        register_spmv(smax, "tmp <- A*p_old", A.get(), p_old, N, tmp, N);
        if (preconditioner == PrecondType::GaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual", L.get(), z_old, N, residual, N);
            register_sptrsv(smax, "M^{-1} * residual", L.get(), z_new, N, residual_new, N);
        } else if (preconditioner == PrecondType::BackwardsGaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual", U.get(), z_old, N, residual, N, true);
            register_sptrsv(smax, "M^{-1} * residual", U.get(), z_new, N, residual_new, N, true);
        } else if (preconditioner == PrecondType::SymmetricGaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual_lower", L.get(), tmp, N, residual, N);
            register_sptrsv(smax, "init M^{-1} * residual_upper", U.get(), z_old, N, tmp, N, true);
            register_sptrsv(smax, "M^{-1} * residual_lower", L.get(), tmp, N, residual_new, N);
            register_sptrsv(smax, "M^{-1} * residual_upper", U.get(), z_new, N, tmp, N, true);
        } else if (preconditioner == PrecondType::TwoStageGS) {
            register_spmv(smax, "init M^{-1} * residual", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * residual", L_strict.get(), work, N, tmp, N);
        } else if (preconditioner == PrecondType::SymmetricTwoStageGS) {
            register_spmv(smax, "init M^{-1} * residual_lower", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "init M^{-1} * residual_upper", U_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * residual_lower", L_strict.get(), work, N, tmp, N);
            register_spmv(smax, "M^{-1} * residual_upper", U_strict.get(), work, N, tmp, N);
        }
    }
#endif

    ~ConjugateGradientSolver() override {
        delete[] x_new;
        delete[] x_old;
        delete[] p_new;
        delete[] p_old;
        delete[] residual_new;
        delete[] residual_old;
        delete[] z_new;
        delete[] z_old;
    }
};
// clang-format on
