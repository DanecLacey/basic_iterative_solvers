#pragma once

#ifdef USE_SMAX
#include "../common.hpp"
#include "SmaxKernels/interface.hpp"

void register_spmv(Interface *smax, const char *kernel_name,
                   const MatrixCRS *crs_mat, double *x, int x_size, double *y,
                   int y_size) {

    smax->register_kernel(kernel_name, SMAX::KernelType::SPMV);
    smax->kernel(kernel_name)
        ->register_A(crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz,
                     crs_mat->col, crs_mat->row_ptr, crs_mat->val);

    // X and Y are dense matrices
    smax->kernel(kernel_name)->register_B(x_size, x);
    smax->kernel(kernel_name)->register_C(y_size, y);
}

void register_sptrsv(Interface *smax, const char *kernel_name,
                     const MatrixCRS *crs_mat, double *x, int x_size, double *y,
                     int y_size, bool is_upper_triang = false) {

    smax->register_kernel(kernel_name, SMAX::KernelType::SPTRSV);
    smax->kernel(kernel_name)
        ->register_A(crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz,
                     crs_mat->col, crs_mat->row_ptr, crs_mat->val);

    // X and Y are dense matrices
    smax->kernel(kernel_name)->register_B(x_size, x);
    smax->kernel(kernel_name)->register_C(y_size, y);

    smax->kernel(kernel_name)->set_mat_upper_triang(is_upper_triang);
}

#endif
