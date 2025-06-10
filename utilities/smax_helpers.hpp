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

    // Tell SMAX to expect a permuted matrix
    // This enables level-set scheduling for SpTRSV
    if (TO_STRING(PERM_MODE) != std::string("NONE"))
        smax->kernel(kernel_name)->set_mat_perm(true);

    smax->kernel(kernel_name)
        ->register_A(crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz,
                     crs_mat->col, crs_mat->row_ptr, crs_mat->val);

    // X and Y are dense matrices
    smax->kernel(kernel_name)->register_B(x_size, x);
    smax->kernel(kernel_name)->register_C(y_size, y);

    smax->kernel(kernel_name)->set_mat_upper_triang(is_upper_triang);
}

void permute_mat(SMAX::Interface *smax, std::unique_ptr<MatrixCRS> &crs_mat) {
    int n_rows = crs_mat->n_rows;
    int n_cols = crs_mat->n_cols;
    int nnz = crs_mat->nnz;
    std::unique_ptr<MatrixCRS> crs_mat_perm =
        std::make_unique<MatrixCRS>(n_rows, n_cols, nnz);

    smax->utils->generate_perm<int>(crs_mat->n_rows, crs_mat->row_ptr,
                                    crs_mat->col, crs_mat->perm,
                                    crs_mat->inv_perm, TO_STRING(PERM_MODE));

    // Apply permutation vector to A
    smax->utils->apply_mat_perm<int, double>(
        crs_mat->n_rows, crs_mat->row_ptr, crs_mat->col, crs_mat->val,
        crs_mat_perm->row_ptr, crs_mat_perm->col, crs_mat_perm->val,
        crs_mat->perm, crs_mat->inv_perm);

    crs_mat = std::move(crs_mat_perm);
}

#endif
