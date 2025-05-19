#ifndef RICHARDSON_HPP
#define RICHARDSON_HPP

#include "../common.hpp"
#include "../kernels.hpp"
#include "../sparse_matrix.hpp"

void richardson_separate_iteration(Timers *timers, const MatrixCRS *crs_mat,
                                   const double *b, double *x_new,
                                   const double *x_old, double *tmp,
                                   double *residual, const double alpha,
                                   Interface *smax = nullptr) {

    int N = crs_mat->n_cols;

    // Update the residual r <- b - A*x
    TIME(timers->spmv,
         spmv(crs_mat, x_new, tmp SMAX_ARGS(0, smax, "update_residual")))

    TIME(timers->sum, subtract_vectors(residual, b, tmp, N))

    // x_new <- x_old + alpha * r
    TIME(timers->sum, sum_vectors(x_new, x_old, residual, N, alpha))
}

#endif