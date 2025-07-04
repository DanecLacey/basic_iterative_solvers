#pragma once
#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

// clang-format off
void cg_separate_iteration(Timers *timers, Solver* solver, double *x_new, double *x_old, double *tmp, double *p_new, double *p_old, double *r_new, double *r_old, double *z_new, double *z_old, Interface *smax = nullptr) {
    int N = solver->crs_mat->n_cols;
    TIME(timers->spmv, spmv(solver->crs_mat.get(), p_old, tmp, 0, smax, "tmp <- A*p_old"));
    double tmp_dot;
    TIME(timers->dot, tmp_dot = dot(r_old, z_old, N));
    double alpha;
    TIME(timers->dot, alpha = tmp_dot / dot(tmp, p_old, N));
    TIME(timers->sum, sum_vectors(x_new, x_old, p_old, N, alpha));
    TIME(timers->sum, subtract_vectors(r_new, r_old, tmp, N, alpha));
    TIME(timers->precond, {
        if (solver->preconditioner == PrecondType::ILU0) {
            apply_preconditioner(solver->L_factor.get(), solver->U_factor.get(), solver->D_factor_vals.get(), z_new, r_new, tmp);
        } else {
            apply_preconditioner(solver->preconditioner, N, solver->crs_mat_L_strict.get(), solver->crs_mat_U_strict.get(), solver->D, z_new, r_new, tmp, 0, smax, "M^{-1} * residual");
        }
    });
    double beta;
    TIME(timers->dot, beta = dot(r_new, z_new, N) / tmp_dot);
    TIME(timers->sum, sum_vectors(p_new, z_new, p_old, N, beta));
}
class ConjugateGradientSolver : public Solver {
  public:
    double *x_new, *x_old;
    double *p_new, *p_old;
    double *z_new, *z_old;
    double *residual_old, *residual_new;

    ConjugateGradientSolver(const Args *cli_args) : Solver(cli_args) {}

    void allocate_structs(const int N) override {
        Solver::allocate_structs(N);
        x_new = new double[N]; x_old = new double[N];
        p_new = new double[N]; p_old = new double[N];
        residual_new = new double[N]; residual_old = new double[N];
        z_new = new double[N]; z_old = new double[N];
    }

    void init_structs(const int N) override {
        Solver::init_structs(N);
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            x_new[i] = 0.0; x_old[i] = x_0[i];
            p_new[i] = 0.0; p_old[i] = 0.0;
            residual_new[i] = 0.0; residual_old[i] = 0.0;
            z_new[i] = 0.0; z_old[i] = 0.0;
        }
    }

     void init_residual() override {
        // The call to compute_residual must be conditional
#ifdef USE_SMAX
        compute_residual(crs_mat.get(), x_old, b, residual, tmp, this->smax, "residual_spmv");
#else
        compute_residual(crs_mat.get(), x_old, b, residual, tmp);
#endif
        
        if (preconditioner == PrecondType::ILU0) {
            apply_preconditioner(L_factor.get(), U_factor.get(), D_factor_vals.get(), z_old, residual, tmp);
        } else {
#ifdef USE_SMAX
            apply_preconditioner(preconditioner, crs_mat->n_cols, crs_mat_L_strict.get(), crs_mat_U_strict.get(), D, z_old, residual, tmp, 0, this->smax, "init M^{-1} * residual");
#else
            apply_preconditioner(preconditioner, crs_mat->n_cols, crs_mat_L_strict.get(), crs_mat_U_strict.get(), D, z_old, residual, tmp);
#endif
        }
        
        copy_vector(p_old, z_old, crs_mat->n_cols);
        copy_vector(residual_old, residual, crs_mat->n_cols);
        residual_norm = euclidean_vec_norm(residual, crs_mat->n_cols);
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
#ifdef USE_SMAX
        // If SMAX is defined, call the helper and pass the smax pointer.
        cg_separate_iteration(timers, this, x_new, x_old, tmp, p_new,
                              p_old, residual_new, residual_old, z_new,
                              z_old, this->smax);
#else
        // If SMAX is NOT defined, call the helper without the last argument.
        // The default `nullptr` will be used.
        cg_separate_iteration(timers, this, x_new, x_old, tmp, p_new,
                              p_old, residual_new, residual_old, z_new,
                              z_old);
#endif
    }
    

    void exchange() override {
        std::swap(p_old, p_new);
        std::swap(z_old, z_new);
        std::swap(residual_old, residual_new);
        std::swap(x_old, x_new);
    }

    void save_x_star() override {
        std::swap(x_old, x_star);
        Solver::save_x_star();
    }

    void record_residual_norm() override {
        residual_norm = euclidean_vec_norm(residual_new, crs_mat->n_cols);
        Solver::record_residual_norm();
    }

    ~ConjugateGradientSolver() override {
        delete[] x_new; delete[] x_old;
        delete[] p_new; delete[] p_old;
        delete[] residual_new; delete[] residual_old;
        delete[] z_new; delete[] z_old;
    }
};
// clang-format on
