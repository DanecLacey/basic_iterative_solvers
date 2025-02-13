#ifndef JACOBI_HPP
#define JACOBI_HPP

#include "../kernels.hpp"
#include "../sparse_matrix.hpp"

void jacobi_fused_iteration(
	MatrixCRS *crs_mat,
	double *b,
	double *x_old,
	double *x_new
){
	double diag_elem;

	#pragma omp parallel for
	for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
			double sum = 0.0;
			for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
					if(row_idx == crs_mat->col[nz_idx]){
							diag_elem = crs_mat->val[nz_idx];
					}
					else{
							sum += crs_mat->val[nz_idx] * x_old[crs_mat->col[nz_idx]];
					}
			}
			x_new[row_idx] = (b[row_idx] - sum) / diag_elem;
	}
}

void normalize_x(
	double *x_new,
	const double *x_old,
	const double *D,
	const double *b,
	int n_rows
){
	double scaled_x_old;
	double adjusted_x;

	#pragma omp parallel for
	for(int row_idx = 0; row_idx < n_rows; ++row_idx){
		scaled_x_old = D[row_idx] * x_old[row_idx];

		adjusted_x = x_new[row_idx] - scaled_x_old;

		x_new[row_idx] = (b[row_idx] - adjusted_x) / D[row_idx];
	}
}

void jacobi_separate_iteration(
	MatrixCRS *crs_mat,
	double *D,
	double *b,
	double *x_old,
	double *x_new
){
	// x_new <- Ax_old
	spmv(crs_mat, x_old, x_new);

	// x_new <- D^{-1}(b - (x_new - D))
	normalize_x(x_new, x_old, D, b, crs_mat->n_rows);
}

#endif