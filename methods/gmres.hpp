#pragma once

#include "../solver.hpp"
#include "../utilities/smax_helpers.hpp"

void orthogonalize_V(Timers *timers, int N, int n_solver_iters, int restart_len,
                     double *H, double *V, double *w) {

    // For all v \in V
    for (int j = 0; j <= n_solver_iters; ++j) {

        // h_ij <- (w_j, v_i)
        TIME(timers->dot,
             H[n_solver_iters + j * restart_len] = dot(w, &V[j * N], N))

        IF_DEBUG_MODE_FINE(printf("h_%i_%i = %f\n", j, n_solver_iters,
                                  H[n_solver_iters + j * restart_len]))

        // w_j <- w_j - h_ij*v_k
        TIME(timers->sum, subtract_vectors(w, w, &V[j * N], N,
                                           H[n_solver_iters + j * restart_len]))

        IF_DEBUG_MODE_FINE(
            SanityChecker::print_vector<double>(w, N, "adjusted_w"))
    }

    IF_DEBUG_MODE_FINE(
        SanityChecker::print_dense_mat<double>(V, restart_len, N, "V"))

    // Save norm to Hessenberg matrix subdiagonal H[k+1,k]
    TIME(timers->norm,
         H[(n_solver_iters + 1) * restart_len + n_solver_iters] =
             euclidean_vec_norm(w, N)) // NOTE: usually 2 norm (Saad)

    IF_DEBUG_MODE_FINE(SanityChecker::print_dense_mat<double>(
        H, (restart_len + 1), restart_len, "H"))

    // Normalize the new orthogonal vector v <- v/H[k+1,k]
    TIME(timers->scale,
         scale(&V[(n_solver_iters + 1) * N], w,
               1.0 / H[(n_solver_iters + 1) * restart_len + n_solver_iters], N))

    IF_DEBUG_MODE_FINE(SanityChecker::print_vector<double>(
        &V[(n_solver_iters + 1) * N], N, "v_j"))
    IF_DEBUG_MODE(SanityChecker::check_V_orthonormal(V, n_solver_iters, N))
}

void least_squares(Timers *timers, int N, int n_solver_iters, int restart_len,
                   double *J, double *H, double *H_tmp, double *Q,
                   double *Q_tmp, double *R) {
    // Per-iteration "local" Givens rotation (m+1 x m+1) matrix
    TIME(timers->copy1,
         init_dense_identity_matrix(J, (restart_len + 1), (restart_len + 1)))

    // The effect all of rotations so far upon the (k+1 x k) Hesseberg matrix
    // (except we store as row vectors, for ...reasons)
    TIME(timers->copy1,
         init_dense_identity_matrix(H_tmp, (restart_len + 1), restart_len))

    IF_DEBUG_MODE_FINE(SanityChecker::print_dense_mat<double>(
        H_tmp, (restart_len + 1), restart_len, "H_tmp_old"))
    if (n_solver_iters == 0) {
        TIME(timers->copy1,
             copy_dense_matrix(H_tmp, H, (restart_len + 1), restart_len))
    } else {
        // H_tmp <- Q*H [(m+1 x m) = (m+1 x m+1)(m+1 x m)] i.e. perform all
        // rotations on H
        TIME(timers->dgemm, dgemm_transpose2(Q, H, H_tmp, (restart_len + 1),
                                             (restart_len + 1), restart_len))
    }
    IF_DEBUG_MODE_FINE(SanityChecker::print_dense_mat<double>(
        H_tmp, (restart_len + 1), restart_len, "H_tmp_new"))

    // clang-format off
    // Form Givens rotation matrix for next iteration
    double J_denom = std::sqrt(
        std::pow(H_tmp[(n_solver_iters * restart_len) + n_solver_iters], 2) +
        std::pow(H_tmp[(n_solver_iters + 1) * restart_len + n_solver_iters],2)
    );

    double c_i = H_tmp[n_solver_iters * restart_len + n_solver_iters] / J_denom;
    double s_i = H_tmp[((n_solver_iters + 1) * restart_len) + n_solver_iters] / J_denom;

    J[n_solver_iters * (restart_len + 1) + n_solver_iters] = c_i; // J[0][0] locally
    J[n_solver_iters * (restart_len + 1) + (n_solver_iters + 1)] = s_i; // J[0][1] locally
    J[(n_solver_iters + 1) * (restart_len + 1) + n_solver_iters] = -1.0 * s_i; // J[1][0] locally
    J[(n_solver_iters + 1) * (restart_len + 1) + (n_solver_iters + 1)] = c_i; // J[1][1] locally
    // clang-format on

    IF_DEBUG_MODE_FINE(SanityChecker::print_dense_mat<double>(
        J, (restart_len + 1), (restart_len + 1), "J"))

    // Q_tmp <- J*Q [(m+1 x m+1) = (m+1 x m+1)(m+1 x m+1)]
    // i.e. Combine local Givens rotations with all previous rotations
    IF_DEBUG_MODE_FINE(SanityChecker::print_dense_mat<double>(
        Q, (restart_len + 1), (restart_len + 1), "Q_old"))

    TIME(timers->dgemm, dgemm_transpose2(J, Q, Q_tmp, (restart_len + 1),
                                         (restart_len + 1), (restart_len + 1)))

    // Q <- Q_copy
    TIME(timers->copy1,
         copy_dense_matrix(Q, Q_tmp, (restart_len + 1), (restart_len + 1)))
    IF_DEBUG_MODE_FINE(SanityChecker::print_dense_mat<double>(
        Q, (restart_len + 1), (restart_len + 1), "Q_new"))

    // R <- Q*H [(m+1 x m) <- (m+1 x m+1)(m+1 x m)]
    TIME(timers->dgemm, dgemm_transpose2(Q, H, R, (restart_len + 1),
                                         (restart_len + 1), restart_len))

    IF_DEBUG_MODE_FINE(SanityChecker::print_dense_mat<double>(
        R, (restart_len + 1), restart_len, "R"))
    IF_DEBUG_MODE(SanityChecker::check_H(H, R, Q, restart_len))
}

void update_g(Timers *timers, int N, int n_solver_iters, int restart_len,
              double *Q, double *g, double *g_tmp, double &residual_norm,
              double beta) {
    IF_DEBUG_MODE_FINE(
        SanityChecker::print_vector<double>(g, (restart_len + 1), "g_old"))
    TIME(timers->copy2, init_vector(g_tmp, 0.0, (restart_len + 1)))
    g_tmp[0] = beta;
    TIME(timers->copy2, copy_vector(g, g_tmp, (restart_len + 1)))

    // TODO: Replace dgemm w/ appropriate dgemv
    // g_k+1 <- Q*g_k [(m+1 x 1) = (m+1 x m+1)(m+1 x 1)]
    TIME(timers->dgemv,
         dgemv(Q, g, g_tmp, (restart_len + 1), (restart_len + 1)))
    // TIME(timers->dgemv, dgemm(Q, g, g_tmp, (restart_len+1), (restart_len+1),
    // 1))

    // g_k+1 <- g_tmp
    TIME(timers->copy2, copy_vector(g, g_tmp, (restart_len + 1)))
    IF_DEBUG_MODE_FINE(
        SanityChecker::print_vector<double>(g, (restart_len + 1), "g_new"))

    // Extract the last element from g as residual norm
    residual_norm = std::abs(g[n_solver_iters + 1]);
    IF_DEBUG_MODE_FINE(printf("GMRES residual_norm on iteration %i: %f\n",
                              n_solver_iters, residual_norm))
}

void gmres_separate_iteration(
    Timers *timers, const PrecondType preconditioner, const MatrixCRS *crs_mat,
    const MatrixCRS *crs_mat_L, const MatrixCRS *crs_mat_U, double *D,
    int n_solver_iters, const int restart_count, const int restart_len,
    double &residual_norm, double *V, double *H, double *H_tmp, double *J,
    double *Q, double *Q_tmp, double *w, double *R, double *g, double *g_tmp,
    double *b, double *x, double *tmp, double beta, Interface *smax = nullptr) {
    /* NOTES:
            - The orthonormal vectors in V are stored as row vectors
    */

    n_solver_iters -= restart_count * restart_len;
    int N = crs_mat->n_cols;
    IF_DEBUG_MODE(
        SanityChecker::print_gmres_iter_counts(n_solver_iters, restart_count))

    // w_j <- A*v_j
    TIME(timers->spmv,
         spmv(crs_mat, &V[n_solver_iters * N],
              w SMAX_ARGS(n_solver_iters * N, smax, "w_j <- A*v_j")))

    // w_j <- M^{-1}w_j
    TIME(timers->precond,
         apply_preconditioner(preconditioner, crs_mat_L, crs_mat_U, D, w, w,
                              tmp SMAX_ARGS(0, smax, "M^{-1} * w_j")))

    IF_DEBUG_MODE_FINE(SanityChecker::print_vector<double>(w, N, "w"))

    TIME(timers->orthog,
         orthogonalize_V(timers, N, n_solver_iters, restart_len, H, V, w))

    TIME(timers->least_sq, least_squares(timers, N, n_solver_iters, restart_len,
                                         J, H, H_tmp, Q, Q_tmp, R))

    TIME(timers->update_g, update_g(timers, N, n_solver_iters, restart_len, Q,
                                    g, g_tmp, residual_norm, beta))
}

class GMRESSolver : public Solver {
  public:
    // Solver-specific fields
    double *x = nullptr;
    double *x_old = nullptr;
    double *V = nullptr;
    double *Vy = nullptr;
    double *y = nullptr;
    double *H = nullptr;
    double *H_tmp = nullptr;
    double *J = nullptr;
    double *Q = nullptr;
    double *Q_tmp = nullptr;
    double *w = nullptr;
    double *R = nullptr;
    double *g = nullptr;
    double *g_tmp = nullptr;
    double beta = 0.0;

    GMRESSolver(const Args *cli_args) : Solver(cli_args) {
        // GMRES-specific initialization?
    }

    void allocate_structs() override {
        Solver::allocate_structs();
        x = new double[crs_mat->n_cols];
        x_old = new double[crs_mat->n_cols];
        V = new double[crs_mat->n_cols * (gmres_restart_len + 1)];
        Vy = new double[crs_mat->n_cols];
        y = new double[gmres_restart_len];
        H = new double[(gmres_restart_len + 1) * gmres_restart_len];
        H_tmp = new double[(gmres_restart_len + 1) * gmres_restart_len];
        J = new double[(gmres_restart_len + 1) * (gmres_restart_len + 1)];
        Q = new double[(gmres_restart_len + 1) * (gmres_restart_len + 1)];
        Q_tmp = new double[(gmres_restart_len + 1) * (gmres_restart_len + 1)];
        w = new double[crs_mat->n_cols];
        R = new double[gmres_restart_len * (gmres_restart_len + 1)];
        g = new double[gmres_restart_len + 1];
        g_tmp = new double[gmres_restart_len + 1];
    }

    void init_structs() override {
        Solver::init_structs();
        // NOTE: We only want to copy x <- x_0 on the first invocation of this
        // routine. All other invocations will be due to resets, in which case
        // the approximate x vector will be explicity computed.
        if (!gmres_restarted) {
#pragma omp parallel for
            for (int i = 0; i < crs_mat->n_cols; ++i) {
                x[i] = x_0[i];
                x_old[i] = x_0[i];
            }
        }

#pragma omp parallel for
        for (int i = 0; i < crs_mat->n_cols * (gmres_restart_len + 1); ++i) {
            V[i] = 0.0;
        }

        // TODO: Is there a better way to do this?
        init_vector(Vy, 0.0, crs_mat->n_cols);
        init_vector(w, 0.0, crs_mat->n_cols);
        init_vector(y, 0.0, gmres_restart_len);
        init_vector(g, 0.0, (gmres_restart_len + 1));
        init_vector(g_tmp, 0.0, (gmres_restart_len + 1));
        init_vector(H, 0.0, (gmres_restart_len + 1) * gmres_restart_len);
        init_vector(H_tmp, 0.0, (gmres_restart_len + 1) * gmres_restart_len);
        init_dense_identity_matrix(J, (gmres_restart_len + 1),
                                   (gmres_restart_len + 1));
        init_vector(R, 0.0, (gmres_restart_len + 1) * gmres_restart_len);
        init_dense_identity_matrix(Q, (gmres_restart_len + 1),
                                   (gmres_restart_len + 1));
        init_dense_identity_matrix(Q_tmp, (gmres_restart_len + 1),
                                   (gmres_restart_len + 1));
    }

    void init_residual() override {
        IF_DEBUG_MODE(
            SanityChecker::print_vector(x, crs_mat->n_cols, "old_x1"));
        compute_residual(crs_mat, x, b, residual,
                         tmp SMAX_ARGS(smax, "residual_spmv"));

        // Precondition the initial residual
        IF_DEBUG_MODE(SanityChecker::print_vector(
            residual, crs_mat->n_cols, "residual before preconditioning"));
        apply_preconditioner(preconditioner, crs_mat_L_strict, crs_mat_U_strict,
                             D, residual, residual,
                             tmp SMAX_ARGS(0, smax, "init M^{-1} * residual"));
        IF_DEBUG_MODE(SanityChecker::print_vector(
            residual, crs_mat->n_cols, "residual after preconditioning"));

        IF_DEBUG_MODE(
            SanityChecker::print_vector(x, crs_mat->n_cols, "old_x2"));
        residual_norm = euclidean_vec_norm(residual, crs_mat->n_cols);
        beta = residual_norm; // NOTE: Beta should be according to
                              // euclidean norm (Saad)

        g[0] = beta;
        g_tmp[0] = beta;

        // V[0] <- r / beta
        // i.e. The first row of V (orthonormal search vectors) gets scaled
        // initial residual
        scale(V, residual, 1.0 / beta, crs_mat->n_cols);
        IF_DEBUG_MODE(SanityChecker::print_vector(residual, crs_mat->n_cols,
                                                  "init_residual"));
        IF_DEBUG_MODE(printf("||init_residual||_2 = %f\n", residual_norm))
        IF_DEBUG_MODE(
            SanityChecker::print_vector(V, crs_mat->n_cols, "init_v"));
        Solver::init_residual();
    }

    void iterate(Timers *timers) override {
        gmres_separate_iteration(
            timers, preconditioner, crs_mat, crs_mat_L_strict, crs_mat_U_strict,
            D, iter_count, gmres_restart_count, gmres_restart_len,
            residual_norm, V, H, H_tmp, J, Q, Q_tmp, w, R, g, g_tmp, b, x, tmp,
            beta SMAX_ARGS(smax));
    }

    void get_explicit_x() override {
        // NOTE: Only relevant for GMRES, so we don't worry about other solvers
        double diag_elem = 1.0;

        // Adjust for restarting
        int n_solver_iters = iter_count;
        n_solver_iters -= gmres_restart_count * gmres_restart_len;
        IF_DEBUG_MODE(SanityChecker::print_gmres_iter_counts(
            n_solver_iters, gmres_restart_count))

        // Backward triangular solve y <- R^{-1}(g) [(m+1 x m)(m x 1) = (m+1 x
        // 1)] Traverse R \in \mathbb{R}^(m+1 x m) from last to first row
        for (int row_idx = n_solver_iters - 1; row_idx >= 0; --row_idx) {
            double sum = 0.0;
            for (int col_idx = row_idx; col_idx < gmres_restart_len;
                 ++col_idx) {
                if (row_idx == col_idx) {
                    diag_elem = R[(row_idx * gmres_restart_len) + col_idx];
                } else {
                    sum +=
                        R[(row_idx * gmres_restart_len) + col_idx] * y[col_idx];
                }
            }
            y[row_idx] = (g[row_idx] - sum) / diag_elem;
#ifdef DEBUG_MODE_FINE
            std::cout << g[row_idx] << " - " << sum << " / " << diag_elem
                      << std::endl;
#endif
        }

        // TODO: Change to appropriate dgemv routine
        // Vy <- V*y [(m x 1) = (m x n)(n x 1)]
        // dgemm_transpose1(V, y, Vy, (gmres_restart_len
        // + 1), crs_mat->n_cols, 1);
        dgemm_transpose1(V, y, Vy, crs_mat->n_cols, gmres_restart_len, 1);

        // dense_MMM_t<VT>(V, &y[0], Vy, n_rows, restart_len, 1);

        IF_DEBUG_MODE_FINE(
            SanityChecker::print_vector(Vy, crs_mat->n_cols, "Vy"));

        // Finally, compute x <- x_0 + Vy [(n x 1) = (n x 1) + (n x m)(m x 1)]
        for (int i = 0; i < crs_mat->n_cols; ++i) {
            x[i] = x_old[i] + Vy[i];
#ifdef DEBUG_MODE_FINE
            std::cout << "x[" << i << "] = " << x_old[i] << " + " << Vy[i]
                      << " = " << x[i] << std::endl;
#endif
        }

        IF_DEBUG_MODE_FINE(
            SanityChecker::print_vector(x, crs_mat->n_cols, "new_x"));
    }

    void save_x_star() override {
        get_explicit_x();
        std::swap(x, x_star);
        Solver::save_x_star();
    }

    void record_residual_norm() override {
        // Norm computed implicitly
        Solver::record_residual_norm();
    }

    void check_restart() override {
        // NOTE: Only relevant for GMRES, so we don't worry about other solvers
        bool norm_convergence = residual_norm < stopping_criteria;
        bool over_max_iters = iter_count > max_iters;
        bool restart_cycle_reached =
            ((iter_count) % (gmres_restart_len) == 0) && (iter_count != 0);
        if (!norm_convergence && !over_max_iters && restart_cycle_reached) {
            gmres_restarted = true;

            IF_DEBUG_MODE(printf("GMRES restart: %i\n", gmres_restart_count))
            // x <- x_0 + Vy
            get_explicit_x();
            copy_vector(x_old, x, crs_mat->n_cols);

            // Re-initialize relevant data structures after restarting GMRES
            // NOTE: x is the only struct which is not re-initialized
            init_structs();

            // TODO: This shouldn't be necessary
            // Re-initialize residual with new inital x approximation
            init_residual();

            ++gmres_restart_count;
        }
    }

    // clang-format off

    void exchange() override {
        // Nothing to exchange here
    }

#ifdef USE_SMAX
    void register_structs() override {
        int N = crs_mat->n_cols;
        register_spmv(smax, "residual_spmv", crs_mat, x, N, tmp, N);
        register_spmv(smax, "w_j <- A*v_j", crs_mat, V, N * (gmres_restart_len + 1), w, N);
        if (preconditioner == PrecondType::GaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual", crs_mat_L, residual, N, residual, N);
            register_sptrsv(smax, "M^{-1} * w_j", crs_mat_L, w, N, w, N);
        } else if (preconditioner == PrecondType::BackwardsGaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual", crs_mat_U, residual, N, residual, N, true);
            register_sptrsv(smax, "M^{-1} * w_j", crs_mat_U, w, N, w, N, true);
        } else if (preconditioner == PrecondType::SymmetricGaussSeidel) {
            register_sptrsv(smax, "init M^{-1} * residual_lower", crs_mat_L, tmp, N, residual, N);
            register_sptrsv(smax, "init M^{-1} * residual_upper", crs_mat_U, residual, N, tmp, N, true);
            register_sptrsv(smax, "M^{-1} * w_j_lower", crs_mat_L, tmp, N, w, N);
            register_sptrsv(smax, "M^{-1} * w_j_upper", crs_mat_U, w, N, tmp, N, true);
        }
    }
#endif
    // clang-format on

    ~GMRESSolver() override {
        delete[] x;
        delete[] x_old;
        delete[] V;
        delete[] Vy;
        delete[] y;
        delete[] H;
        delete[] H_tmp;
        delete[] J;
        delete[] Q;
        delete[] Q_tmp;
        delete[] w;
        delete[] R;
        delete[] g;
        delete[] g_tmp;
    }
};
