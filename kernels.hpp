#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "common.hpp"
#include "sparse_matrix.hpp"

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

void spmv(
    const MatrixCRS *crs_mat,
    const double *x,
		double *y
)
{
	#pragma omp parallel
	{
#ifdef USE_LIKWID
		LIKWID_MARKER_START("spmv");
#endif
		#pragma omp for schedule(static)
		for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
			double tmp = 0.0;
			#pragma omp simd simdlen(SIMD_LENGTH) reduction(+:tmp)
			for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
#ifdef DEBUG_MODE_FINE
				printf("nz_idx = %i\n", nz_idx);
				printf("crs_mat->val[nz_idx] = %f\n", crs_mat->val[nz_idx]);
				printf("crs_mat->col[nz_idx] = %i\n", crs_mat->col[nz_idx]);
				printf("x[crs_mat->col[nz_idx]] = %f\n", x[crs_mat->col[nz_idx]]);
#endif
				tmp += crs_mat->val[nz_idx] * x[crs_mat->col[nz_idx]];
			}
			y[row_idx] = tmp;
		}
#ifdef USE_LIKWID
		LIKWID_MARKER_STOP("spmv");
#endif
	}
}

void subtract_vectors(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    int N,
    double scale = 1.0
){
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        result_vec[i] = vec1[i] - scale*vec2[i];
    }
}

void compute_residual(
	const MatrixCRS *crs_mat,
	const double *x,
  const double *b,
  double *residual,
  double *tmp
){
    spmv(crs_mat, x, tmp);
    subtract_vectors(residual, b, tmp, crs_mat->n_cols);
}

double infty_vec_norm(
    const double *vec,
    int N
){
    double max_abs = 0.;
    double curr_abs;
    for (int i = 0; i < N; ++i){
        // TODO:: Hmmm...
        // curr_abs = std::abs(static_cast<double>(vec[i]));
        curr_abs = (vec[i] >= 0) ? vec[i]  : -1*vec[i];
        if ( curr_abs > max_abs){
            max_abs = curr_abs; 
        }
    }

    return max_abs;
}

void spltsv(
    const MatrixCRS *crs_mat_L,
    double *x,
    const double *D,
    const double *b
)
{
#ifdef USE_LIKWID
	LIKWID_MARKER_START("spltsv");
#endif
	double sum;
	for(int row_idx = 0; row_idx < crs_mat_L->n_rows; ++row_idx){
		sum = 0.0;
		for(int nz_idx = crs_mat_L->row_ptr[row_idx]; nz_idx < crs_mat_L->row_ptr[row_idx+1]; ++nz_idx){
			sum += crs_mat_L->val[nz_idx] * x[crs_mat_L->col[nz_idx]];
		}
		x[row_idx] = (b[row_idx] - sum) / D[row_idx];
	}
#ifdef USE_LIKWID
	LIKWID_MARKER_STOP("spltsv");
#endif
}

#endif
