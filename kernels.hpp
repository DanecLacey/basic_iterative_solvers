#ifndef KERNELS_HPP
#define KERNELS_HPP

#ifndef PRECOND_ITERS
#define PRECOND_ITERS 1
#endif

#include "common.hpp"
#include "sparse_matrix.hpp"

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

void spmv(
    const MatrixCRS *crs_mat,
    const double *x,
    double *y)
{
#pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv");
#endif
#pragma omp for schedule(static)
        for (int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx)
        {
            double tmp = 0.0;
#pragma omp simd reduction(+ : tmp)
            for (int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx + 1]; ++nz_idx)
            {
                // #ifdef DEBUG_MODE_FINE
                // 				printf("nz_idx = %i\n", nz_idx);
                // 				printf("crs_mat->val[nz_idx] = %f\n", crs_mat->val[nz_idx]);
                // 				printf("crs_mat->col[nz_idx] = %i\n", crs_mat->col[nz_idx]);
                // 				printf("x[crs_mat->col[nz_idx]] = %f\n", x[crs_mat->col[nz_idx]]);
                // #endif
                tmp += crs_mat->val[nz_idx] * x[crs_mat->col[nz_idx]];
            }
            y[row_idx] = tmp;
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv");
#endif
    }
}

// TODO: JH
// // Saad: Iterative Methods for Sparse Linear Systems (ch 11.6)
// void spltsv_lvl(

// ){
//     // TODO
// }

// // https://icl.utk.edu/files/publications/2018/icl-utk-1067-2018.pdf
// // https://www.nrel.gov/docs/fy22osti/80263.pdf
// void spltsv_2stage(

// ){
//     // TODO
// }

// // Saad: Iterative Methods for Sparse Linear Systems (ch 12.4.3)
// void spltsv_mc(

// ){
//     // TODO
// }

void spltsv(
    const MatrixCRS *crs_mat_L,
    double *x,
    const double *D,
    const double *b,
    bool warmup = false)
{
#ifdef USE_LIKWID
    if (!warmup)
    {
        LIKWID_MARKER_START("spltsv");
    }
#endif

    double sum;
    for (int row_idx = 0; row_idx < crs_mat_L->n_rows; ++row_idx)
    {
        sum = 0.0;
        for (int nz_idx = crs_mat_L->row_ptr[row_idx]; nz_idx < crs_mat_L->row_ptr[row_idx + 1]; ++nz_idx)
        {
            sum += crs_mat_L->val[nz_idx] * x[crs_mat_L->col[nz_idx]];
        }
        x[row_idx] = (b[row_idx] - sum) / D[row_idx];
    }

#ifdef USE_LIKWID
    if (!warmup)
    {
        LIKWID_MARKER_STOP("spltsv");
    }
#endif
}

void backwards_spltsv(
    const MatrixCRS *crs_mat_U,
    double *x,
    const double *D,
    const double *b)
{
#ifdef USE_LIKWID
    LIKWID_MARKER_START("spltsv");
#endif
    for (int row_idx = crs_mat_U->n_rows - 1; row_idx >= 0; --row_idx)
    {
        double sum = 0.0;
        for (int nz_idx = crs_mat_U->row_ptr[row_idx]; nz_idx < crs_mat_U->row_ptr[row_idx + 1]; ++nz_idx)
        {
            sum += crs_mat_U->val[nz_idx] * x[crs_mat_U->col[nz_idx]];
        }
        x[row_idx] = (b[row_idx] - sum) / D[row_idx];
    }
#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("spltsv");
#endif
}

void subtract_vectors(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    const int N,
    const double scale = 1.0)
{
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        result_vec[i] = vec1[i] - scale * vec2[i];
    }
}

void sum_vectors(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    const int N,
    const double scale = 1.0)
{
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        result_vec[i] = vec1[i] + scale * vec2[i];
    }
}

void elemwise_mult_vectors(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    const int N,
    const double scale = 1.0)
{
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        result_vec[i] = vec1[i] * scale * vec2[i];
    }
}

void elemwise_div_vectors(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    const int N,
    const double scale = 1.0)
{
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        result_vec[i] = vec1[i] / (scale * vec2[i]);
    }
}

void compute_residual(
    const MatrixCRS *crs_mat,
    const double *x,
    const double *b,
    double *residual,
    double *tmp)
{
    spmv(crs_mat, x, tmp);
    subtract_vectors(residual, b, tmp, crs_mat->n_cols);
}

double infty_vec_norm(
    const double *vec,
    const int N)
{
    double max_abs = 0.0;
    double curr_abs;
    for (int i = 0; i < N; ++i)
    {
        // TODO:: Hmmm...
        // curr_abs = std::abs(static_cast<double>(vec[i]));
        curr_abs = (vec[i] >= 0) ? vec[i] : -1 * vec[i];
        if (curr_abs > max_abs)
        {
            max_abs = curr_abs;
        }
    }

    return max_abs;
}

double infty_mat_norm(
    const MatrixCRS *crs_mat)
{
    double max_row_sum = 0.0;

#pragma omp parallel for reduction(max : max_row_sum)
    for (int row = 0; row < crs_mat->n_rows; ++row)
    {
        double row_sum = 0.0;
        for (int idx = crs_mat->row_ptr[row]; idx < crs_mat->row_ptr[row + 1]; ++idx)
        {
            row_sum += std::abs(crs_mat->val[idx]);
        }
        max_row_sum = std::max(max_row_sum, row_sum);
    }

    return max_row_sum;
}

double euclidean_vec_norm(
    const double *vec,
    int N)
{
    double tmp = 0.0;

#pragma omp parallel for reduction(+ : tmp)
    for (int i = 0; i < N; ++i)
    {
        tmp += vec[i] * vec[i];
    }

    return std::sqrt(tmp);
}

double dot(
    const double *vec1,
    const double *vec2,
    const int N)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; ++i)
    {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

void scale(
    double *result_vec,
    const double *vec,
    const double scalar,
    const int N)
{
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        result_vec[i] = vec[i] * scalar;
    }
}

void init_dense_identity_matrix(
    double *mat,
    const int n_rows,
    const int n_cols)
{
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i)
    {
        for (int j = 0; j < n_cols; ++j)
        {
            if (i == j)
            {
                mat[n_cols * i + j] = 1.0;
            }
            else
            {
                mat[n_cols * i + j] = 0.0;
            }
        }
    }
}

void init_vector(
    double *vec,
    double val,
    long size)
{
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        vec[i] = val;
    }
}

void copy_dense_matrix(
    double *new_mat,
    const double *old_mat,
    const int n_rows,
    const int n_cols)
{
    for (int row_idx = 0; row_idx < n_rows; ++row_idx)
    {
        for (int col_idx = 0; col_idx < n_cols; ++col_idx)
        {
            new_mat[n_cols * row_idx + col_idx] = old_mat[n_cols * row_idx + col_idx];
        }
    }
}

void copy_vector(
    double *new_vec,
    const double *old_vec,
    const int n_rows)
{
    for (int row_idx = 0; row_idx < n_rows; ++row_idx)
    {
        new_vec[row_idx] = old_vec[row_idx];
    }
}

void dgemm_transpose1(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B)
{
    for (int i = 0; i < n_rows_A; ++i)
    {
        for (int j = 0; j < n_cols_B; ++j)
        {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k)
            {
                tmp += A[k * n_rows_A + i] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

void dgemm_transpose2(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B)
{
    for (int i = 0; i < n_rows_A; ++i)
    {
        for (int j = 0; j < n_cols_B; ++j)
        {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k)
            {
                tmp += A[i * n_cols_A + k] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

void dgemm(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B)
{
    for (int i = 0; i < n_rows_A; ++i)
    {
        for (int j = 0; j < n_cols_B; ++j)
        {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k)
            {
                tmp += A[i * n_cols_A + k] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

void dgemv(
    const double *A,
    const double *x,
    double *y,
    int n_rows_A,
    int n_cols_A,
    double alpha = 1.0
    // double beta = 1.0
)
{
    for (int i = 0; i < n_rows_A; ++i)
    {
        // y[i] *= beta;
        y[i] = 0.0;
        for (int j = 0; j < n_cols_A; ++j)
        {
            y[i] += alpha * A[i * n_cols_A + j] * x[j];
        }
    }
}

// Computes z <- M^{-1}y
void apply_preconditioner(
    const std::string preconditioner_type,
    const MatrixCRS *crs_mat_L,
    const MatrixCRS *crs_mat_U,
    double *D,
    double *vec,
    double *rhs,
    double *tmp)
{
    int N = crs_mat_L->n_cols;

    for (int i = 0; i < PRECOND_ITERS; ++i)
    {
        if (preconditioner_type == "jacobi")
        {
            elemwise_div_vectors(vec, rhs, D, N);
        }
        else if (preconditioner_type == "gauss-seidel")
        {
            spltsv(crs_mat_L, vec, D, rhs);
        }
        else if (preconditioner_type == "backwards-gauss-seidel")
        {
            backwards_spltsv(crs_mat_U, vec, D, rhs);
        }
        else if (preconditioner_type == "symmetric-gauss-seidel")
        {
            // tmp <- (L+D)^{-1}*r
            spltsv(crs_mat_L, tmp, D, rhs);

            // tmp <- D(L+D)^{-1}*r
            elemwise_mult_vectors(tmp, tmp, D, N);

            // z <- (L+U)^{-1}*tmp
            backwards_spltsv(crs_mat_U, vec, D, tmp);
        }
        else if (preconditioner_type == "ffbb-gauss-seidel")
        {
            // TODO
            spltsv(crs_mat_L, tmp, D, rhs);
            spltsv(crs_mat_L, tmp, D, rhs);

            elemwise_mult_vectors(tmp, tmp, D, N);
            elemwise_mult_vectors(tmp, tmp, D, N);

            backwards_spltsv(crs_mat_U, vec, D, tmp);
            backwards_spltsv(crs_mat_U, vec, D, tmp);
        }
        else
        {
            copy_vector(vec, rhs, N);
        }
    }
}

#endif
