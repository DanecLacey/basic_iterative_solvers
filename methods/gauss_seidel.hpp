#ifndef GAUSS_SEIDEL_HPP
#define GAUSS_SEIDEL_HPP

#include "../kernels.hpp"
#include "../sparse_matrix.hpp"

void gs_fused_iteration(
	MatrixCRS *crs_mat,
  double *tmp,
  double *b,
  double *x
){
    double diag_elem = 1.0;

    for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
			double sum = 0.0;
			for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
				if(row_idx == crs_mat->col[nz_idx]){
					diag_elem = crs_mat->val[nz_idx];
				}
				else{
					sum += crs_mat->val[nz_idx] * x[crs_mat->col[nz_idx]];
				}
			}
			x[row_idx] = (b[row_idx] - sum) / diag_elem;
    }
}

void gs_separate_iteration(
	MatrixCRS *crs_mat_U,
	MatrixCRS *crs_mat_L, 
	double *tmp,
	double *D, 
	double *b, 
	double *x
){
	// tmp <- Ux
	spmv(crs_mat_U, x, tmp);

	// tmp <- b - tmp
	subtract_vectors(tmp, b, tmp, crs_mat_U->n_rows);

	// x <- (D+L)^{-1}(tmp)
	spltsv(crs_mat_L, x, D, tmp);
}

#endif
