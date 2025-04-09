#ifndef VALIDATE_HPP
#define VALIDATE_HPP

#include "bench_macros.hpp"
#include "../kernels.hpp"
#include "../common.hpp"
#include "../utilities.hpp"

#ifdef BENCH_MKL
    #include <mkl.h>
#endif

int validate(int argc, char *argv[]){
#ifdef BENCH_MKL
    Args *cli_args = new Args;
	parse_cli(cli_args, argc, argv, true);

	MatrixCOO *coo_mat = new MatrixCOO;
    READ(coo_mat, cli_args);

	IF_DEBUG_MODE(printf("Converting COO matrix to CRS\n"))
	MatrixCRS *crs_mat = new MatrixCRS;
	convert_coo_to_crs(coo_mat, crs_mat);

	double *D; double *x; double* x_mkl; double *b;
    D =     new double[(crs_mat)->n_cols];
    x =     new double[(crs_mat)->n_cols];
    x_mkl = new double[(crs_mat)->n_cols];
    b =     new double[(crs_mat)->n_cols];

    for (int i = 0; i < crs_mat->n_cols; ++i) {
        x[i] = INIT_X_VAL;
        x_mkl[i] = INIT_X_VAL;
        b[i] = B_VAL;
        D[i] = 0.0;
    }

	IF_DEBUG_MODE(printf("Extracting L, U, and D\n"))
	MatrixCOO *coo_mat_L = new MatrixCOO;
    MatrixCOO *coo_mat_L_plus_D = new MatrixCOO;
	MatrixCOO *coo_mat_U = new MatrixCOO;
	extract_L_U(coo_mat, coo_mat_L, coo_mat_U);
    extract_L_plus_D(coo_mat, coo_mat_L_plus_D);
	extract_D(coo_mat, D);

	IF_DEBUG_MODE(printf("Converting L to CRS\n"))
	MatrixCRS *crs_mat_L = new MatrixCRS;
	convert_coo_to_crs(coo_mat_L, crs_mat_L);
	MatrixCRS *crs_mat_L_plus_D = new MatrixCRS;
	convert_coo_to_crs(coo_mat_L_plus_D, crs_mat_L_plus_D);

    // Get MKL result
    sparse_matrix_t L;
    mkl_sparse_d_create_csr(
        &L, 
        SPARSE_INDEX_BASE_ZERO,
        crs_mat_L_plus_D->n_rows,
        crs_mat_L_plus_D->n_cols, 
        crs_mat_L_plus_D->row_ptr,
        crs_mat_L_plus_D->row_ptr + 1,
        crs_mat_L_plus_D->col,
        crs_mat_L_plus_D->val
    );

    // Set matrix descriptor for lower triangular
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    sparse_status_t status = mkl_sparse_d_trsv(
        SPARSE_OPERATION_NON_TRANSPOSE, 1.0, L, descr, b, x_mkl
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        printf("MKL trsv failed with status %d\n", status);
    }

    // Get our result
    spltsv(crs_mat_L, x, D, b);

    // Compare results
    // TODO: parallelize w/ reduction(max)
    double max_diff = 0.0;
    int x_idx;
    for (x_idx = 0; x_idx < crs_mat->n_cols; ++x_idx) {
        double diff = std::abs(x[x_idx] - x_mkl[x_idx]);
#ifdef DEBUG_MODE_FINE
        printf("x[%d] = %f, x_mkl[%d] = %f, diff = %f\n", x_idx, x[x_idx], x_idx, x_mkl[x_idx], diff);
#endif
        if (diff > max_diff) {
            max_diff = diff;
            if (max_diff > TOL) {
                break;
            }
        }
    }

    delete cli_args;       
    delete coo_mat;        
    delete crs_mat;        
    delete coo_mat_L;      
    delete coo_mat_U;      
    delete crs_mat_L;      
    delete[] D;            
    delete[] x;            
    delete[] b;
    
    std::cout << "Max difference: " << max_diff << std::endl;
    if (max_diff > TOL) {
        std::cout << "Validation failed at x index: " << x_idx << std::endl;
        return 1;
    } else {
        std::cout << "Validation passed!" << std::endl;
        return 0;
    }
#else
    printf("MKL is not enabled. Please compile with -DBENCH_MKL to enable validation.\n");
    return 1;
#endif
}
#endif