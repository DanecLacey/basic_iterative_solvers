#ifndef CG_HPP
#define CG_HPP

#include "../common.hpp"
#include "../kernels.hpp"
#include "../sparse_matrix.hpp"

void cg_separate_iteration(
	Timers *timers,
	MatrixCRS *crs_mat,
	double *x_new,
	double *x_old,
	double *tmp,
	double *p_new,
	double *p_old,
	double *residual_new,
	double *residual_old
){
	// pre-compute tmp <- Ap_old
	TIME(timers->spmv, spmv(crs_mat, p_old, tmp))

	TIME(timers->dot, double r_old_squared = dot(residual_old, residual_old, crs_mat->n_cols))

	// alpha <- (r_old, r_old) / (Ap_old, p_old)
	TIME(timers->dot, double alpha = r_old_squared / dot(tmp, p_old, crs_mat->n_cols))
	
	// x_new <- x_old + alpha * p_old
	TIME(timers->sum, sum_vectors(x_new, x_old, p_old, crs_mat->n_cols, alpha))

	// r_new <- r_old - alpha * Ap_old
	TIME(timers->sum, subtract_vectors(residual_new, residual_old, tmp, crs_mat->n_cols, alpha))

	// beta <- (r_new, r_new) / (r_old, r_old)
	TIME(timers->dot, double beta = dot(residual_new, residual_new, crs_mat->n_cols) / r_old_squared)

	// p_new <- r_new + beta * p_old
	TIME(timers->sum, sum_vectors(p_new, residual_new, p_old, crs_mat->n_cols, beta))
}


#endif
