#pragma once

#ifdef USE_SMAX
#include "../common.hpp"
#include "SmaxKernels/interface.hpp"

void register_spmv(Interface *smax, const char *kernel_name, const MatrixCRS *A,
                   double *x, int x_size, double *y, int y_size) {

    smax->register_kernel(kernel_name, SMAX::KernelType::SPMV);
    smax->kernel(kernel_name)
        ->register_A(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);

    // X and Y are dense matrices
    smax->kernel(kernel_name)->register_B(x_size, x);
    smax->kernel(kernel_name)->register_C(y_size, y);
}

void register_sptrsv(Interface *smax, const char *kernel_name,
                     const MatrixCRS *A, double *x, int x_size, double *y,
                     int y_size, bool is_upper_triang = false) {

    smax->register_kernel(kernel_name, SMAX::KernelType::SPTRSV);

    // Tell SMAX to expect a permuted matrix
    // This enables level-set scheduling for SpTRSV
    if (TO_STRING(PERM_MODE) != std::string("NONE"))
        smax->kernel(kernel_name)->set_mat_perm(true);

    smax->kernel(kernel_name)
        ->register_A(A->n_rows, A->n_cols, A->nnz, A->col, A->row_ptr, A->val);

    // X and Y are dense matrices
    smax->kernel(kernel_name)->register_B(x_size, x);
    smax->kernel(kernel_name)->register_C(y_size, y);

    smax->kernel(kernel_name)->set_mat_upper_triang(is_upper_triang);
}

void permute_mat(SMAX::Interface *smax, std::unique_ptr<MatrixCRS> &A) {
    int n_rows = A->n_rows;
    int n_cols = A->n_cols;
    int nnz = A->nnz;
    std::unique_ptr<MatrixCRS> A_perm =
        std::make_unique<MatrixCRS>(n_rows, n_cols, nnz);

    smax->utils->generate_perm<int>(A->n_rows, A->row_ptr, A->col, A->perm,
                                    A->inv_perm, TO_STRING(PERM_MODE));

    // Apply permutation vector to A
    smax->utils->apply_mat_perm<int, double>(
        A->n_rows, A->row_ptr, A->col, A->val, A_perm->row_ptr, A_perm->col,
        A_perm->val, A->perm, A->inv_perm);

    A = std::move(A_perm);
}

#endif
