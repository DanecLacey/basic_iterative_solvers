#ifndef CG_HPP
#define CG_HPP

#include "../common.hpp"
#include "../kernels.hpp"
#include "../sparse_matrix.hpp"

void cg_separate_iteration(
	Timers *timers,
	const std::string preconditioner_type,
	const MatrixCRS *crs_mat,
	const MatrixCRS *crs_mat_L,
	const MatrixCRS *crs_mat_U,
	double *D,
	double *x_new,
	double *x_old,
	double *tmp,
	double *p_new,
	double *p_old,
	double *r_new,
	double *r_old,
	double *z_new,
	double *z_old
){
	// pre-compute tmp <- Ap_old
	TIME(timers->spmv, spmv(crs_mat, p_old, tmp))

	TIME(timers->dot, double tmp_dot = dot(r_old, z_old, crs_mat->n_cols))

	// alpha <- (r_old, z_old) / (Ap_old, p_old)
	TIME(timers->dot, double alpha = tmp_dot / dot(tmp, p_old, crs_mat->n_cols))

	IF_DEBUG_MODE_FINE(printf("alpha = %f\n", alpha))
	
	// x_new <- x_old + alpha * p_old
	TIME(timers->sum, sum_vectors(x_new, x_old, p_old, crs_mat->n_cols, alpha))

	// r_new <- r_old - alpha * Ap_old
	TIME(timers->sum, subtract_vectors(r_new, r_old, tmp, crs_mat->n_cols, alpha))

	// z_new <- M^{-1}r_new
	TIME(timers->precond, apply_preconditioner(preconditioner_type, crs_mat_L, crs_mat_U, D, z_new, r_new, tmp))

	// beta <- (r_new, r_new) / (r_old, r_old)
	TIME(timers->dot, double beta = dot(r_new, z_new, crs_mat->n_cols) / tmp_dot)

	IF_DEBUG_MODE_FINE(printf("beta = %f\n", beta))

	// p_new <- z_new + beta * p_old
	TIME(timers->sum, sum_vectors(p_new, z_new, p_old, crs_mat->n_cols, beta))
}


#endif
