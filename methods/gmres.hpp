#ifndef GMRES_HPP
#define GMRES_HPP

#include "../common.hpp"
#include "../kernels.hpp"
#include "../sparse_matrix.hpp"

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

    // Form Givens rotation matrix for next iteration
    double J_denom = std::sqrt(
        std::pow(H_tmp[(n_solver_iters * restart_len) + n_solver_iters], 2) +
        std::pow(H_tmp[(n_solver_iters + 1) * restart_len + n_solver_iters],
                 2));

    double c_i = H_tmp[n_solver_iters * restart_len + n_solver_iters] / J_denom;
    double s_i =
        H_tmp[((n_solver_iters + 1) * restart_len) + n_solver_iters] / J_denom;

    J[n_solver_iters * (restart_len + 1) + n_solver_iters] =
        c_i; // J[0][0] locally
    J[n_solver_iters * (restart_len + 1) + (n_solver_iters + 1)] =
        s_i; // J[0][1] locally
    J[(n_solver_iters + 1) * (restart_len + 1) + n_solver_iters] =
        -1.0 * s_i; // J[1][0] locally
    J[(n_solver_iters + 1) * (restart_len + 1) + (n_solver_iters + 1)] =
        c_i; // J[1][1] locally

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
#ifdef USE_SMAX
    SMAX::Interface *smax,
#endif
    Timers *timers, const std::string preconditioner_type,
    const MatrixCRS *crs_mat, const MatrixCRS *crs_mat_L,
    const MatrixCRS *crs_mat_U, double *D, int n_solver_iters,
    const int restart_count, const int restart_len, double &residual_norm,
    double *V, double *H, double *H_tmp, double *J, double *Q, double *Q_tmp,
    double *w, double *R, double *g, double *g_tmp, double *b, double *x,
    double *tmp, double beta) {
    /* NOTES:
            - The orthonormal vectors in V are stored as row vectors
    */

    n_solver_iters -= restart_count * restart_len;
    int N = crs_mat->n_cols;
    IF_DEBUG_MODE(
        SanityChecker::print_gmres_iter_counts(n_solver_iters, restart_count))

    // w_j <- A*v_j
    TIME(timers->spmv,
         spmv(
#ifdef USE_SMAX
             smax, "w_j <- A*v_j",
#endif
             crs_mat, &V[n_solver_iters * N], w, n_solver_iters * N))

    // w_j <- M^{-1}w_j
    TIME(timers->precond, apply_preconditioner(preconditioner_type, crs_mat_L,
                                               crs_mat_U, D, w, w, tmp))

    IF_DEBUG_MODE_FINE(SanityChecker::print_vector<double>(w, N, "w"))

    TIME(timers->orthog,
         orthogonalize_V(timers, N, n_solver_iters, restart_len, H, V, w))

    TIME(timers->least_sq, least_squares(timers, N, n_solver_iters, restart_len,
                                         J, H, H_tmp, Q, Q_tmp, R))

    TIME(timers->update_g, update_g(timers, N, n_solver_iters, restart_len, Q,
                                    g, g_tmp, residual_norm, beta))
}

#endif
