#ifndef BICGSTAB_HPP
#define BICGSTAB_HPP

#include "../common.hpp"
#include "../kernels.hpp"
#include "../sparse_matrix.hpp"

void bicgstab_separate_iteration(
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
	double *residual_new,
	double *residual_old,
	double *residual_0,
	double *v,
	double *h,
	double *s,
	double *t,
	double &rho_new,
	double rho_old
){
	int N = crs_mat->n_cols;

	IF_DEBUG_MODE_FINE(SanityChecker::print_bicgstab_vectors(crs_mat->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new, residual_old, residual_0, v, h, s, t, rho_new, rho_old, "before"))

	// v <- A*p_old
	TIME(timers->spmv, spmv(crs_mat, p_old, v))

	// alpha <- rho_old / (r_0, v)
	TIME(timers->dot, double alpha = rho_old / dot(residual_0, v, N))

	IF_DEBUG_MODE_FINE(printf("alpha = %f\n", alpha))

	// h <- x_old + alpha*p_old
	TIME(timers->sum, sum_vectors(h, x_old, p_old, N, alpha))

	// s <- r_old - alpha*v
	TIME(timers->sum, subtract_vectors(s, residual_old, v, N, alpha))

	// TODO: quit if s is small enough

	// t <- A*s
	TIME(timers->spmv, spmv(crs_mat, s, t))

	// omega <- (t,s) / (t,t)
	TIME(timers->dot, double omega = dot(t, s, N) / dot(t, t, N))

	IF_DEBUG_MODE_FINE(printf("omega = %f\n", omega))

	// x_new <- h + omega*s
	TIME(timers->sum, sum_vectors(x_new, h, s, N, omega))

	// r_new <- s - omega*t
	TIME(timers->sum, subtract_vectors(residual_new, s, t, N, omega))

	// TODO: quit if r_new is small enough

	// rho_new = (r_0, r_new)
	TIME(timers->dot, rho_new = dot(residual_0, residual_new, N))

	double beta = (rho_new / rho_old) * (alpha / omega);

	IF_DEBUG_MODE_FINE(printf("beta = %f\n", beta))

	// tmp <- p_old - omega*v
	TIME(timers->sum, subtract_vectors(tmp, p_old, v, N, omega))

	// p_new <- r_new + beta*tmp
	TIME(timers->sum, sum_vectors(p_new, residual_new, tmp, N, beta))

	IF_DEBUG_MODE_FINE(SanityChecker::print_bicgstab_vectors(crs_mat->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new, residual_old, residual_0, v, h, s, t, rho_new, rho_old, "after"))
}

// https://jiechenjiechen.github.io/pub/fbcgs.pdf
void pbicgstab_separate_iteration(
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
	double *residual_new,
	double *residual_old,
	double *residual_0,
	double *v,
	double *h,
	double *s,
	double *s_tmp,
	double *t,
	double *t_tmp,
	double *y,
	double *z,
	double &rho_new,
	double rho_old
){
	int N = crs_mat->n_cols;

	IF_DEBUG_MODE_FINE(SanityChecker::print_bicgstab_vectors(crs_mat->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new, residual_old, residual_0, v, h, s, t, rho_new, rho_old, "before"))

	// y <- M^{-1}p_old
	TIME(timers->precond, apply_preconditioner(crs_mat_L, crs_mat_U, preconditioner_type, y, p_old, D))

	// v <- A*y
	TIME(timers->spmv, spmv(crs_mat, y, v))

	// alpha <- rho_old / (r_0, v)
	TIME(timers->dot, double alpha = rho_old / dot(residual_0, v, N))

	IF_DEBUG_MODE_FINE(printf("alpha = %f\n", alpha))

	// s <- r_old - alpha*v
	TIME(timers->sum, subtract_vectors(s, residual_old, v, N, alpha))

	// s_tmp <- M^{-1}s
	TIME(timers->precond, apply_preconditioner(crs_mat_L, crs_mat_U, preconditioner_type, s_tmp, s, D))

	// z <- A*s_tmp
	TIME(timers->spmv, spmv(crs_mat, s_tmp, z))

	TIME(timers->dot, double omega = dot(z, s, N) / dot(z, z, N))

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

	IF_DEBUG_MODE_FINE(SanityChecker::print_bicgstab_vectors(crs_mat->n_cols, x_new, x_old, tmp, p_new, p_old, residual_new, residual_old, residual_0, v, h, s, t, rho_new, rho_old, "after"))
}

#endif
