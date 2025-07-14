#pragma once

#include <utility>
#ifndef PRECOND_ITERS
#define PRECOND_ITERS 1
#endif
#ifndef PRECOND_INNER_ITERS
#define PRECOND_INNER_ITERS 1
#endif

#include "common.hpp"
#include "sparse_matrix.hpp"

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef USE_SMAX
#include "utilities/smax_helpers.hpp"
#endif

inline void native_spmv(const MatrixCRS *A, const double *x, double *y) {
#pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv");
#endif
#pragma omp for schedule(static)
        for (int row = 0; row < A->n_rows; ++row) {
            double tmp = 0.0;
#pragma omp simd reduction(+ : tmp)
            for (int nz_idx = A->row_ptr[row]; nz_idx < A->row_ptr[row + 1];
                 ++nz_idx) {
                tmp += A->val[nz_idx] * x[A->col[nz_idx]];
            }
            y[row] = tmp;
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv");
#endif
    }
}

inline void spmv(const MatrixCRS *A, const double *x, double *y, int offset = 0,
                 Interface *smax = nullptr,
                 const std::string kernel_name = "") {
#if USE_SMAX
    smax->kernel(kernel_name.c_str())->run(0, offset, 0);
#else
    native_spmv(A, x, y);
#endif
}

inline void native_sptrsv(const MatrixCRS *L, double *x, const double *D,
                          const double *b) {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("sptrsv");
#endif

    double sum;
    for (int row = 0; row < L->n_rows; ++row) {
        sum = 0.0;
        int row_start = L->row_ptr[row];
        int row_stop = L->row_ptr[row + 1];

        for (int nz_idx = row_start; nz_idx < row_stop; ++nz_idx) {
            sum += L->val[nz_idx] * x[L->col[nz_idx]];
        }

        x[row] = (b[row] - sum) / D[row];
    }

#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("sptrsv");
#endif
}

inline void sptrsv(const MatrixCRS *L, double *x, const double *D,
                   const double *b, int offset = 0, Interface *smax = nullptr,
                   const std::string kernel_name = "") {
#ifdef USE_SMAX
    smax->kernel(kernel_name.c_str())->run(0, offset, 0);
#else
    native_sptrsv(L, x, D, b);
#endif
}

inline void native_bsptrsv(const MatrixCRS *U, double *x, const double *D,
                           const double *b) {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("backwards-sptrsv");
#endif
    for (int row = U->n_rows - 1; row >= 0; --row) {
        double sum = 0.0;
        int row_start = U->row_ptr[row];
        int row_stop = U->row_ptr[row + 1];

        for (int nz_idx = row_start; nz_idx < row_stop; ++nz_idx) {
            sum += U->val[nz_idx] * x[U->col[nz_idx]];
        }

        x[row] = (b[row] - sum) / D[row];
    }
#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("backwards-sptrsv");
#endif
}

inline void bsptrsv(const MatrixCRS *U, double *x, const double *D,
                    const double *b, int offset = 0, Interface *smax = nullptr,
                    const std::string kernel_name = "") {
#if USE_SMAX
    smax->kernel(kernel_name.c_str())->run(0, offset, 0);
#else
    native_bsptrsv(U, x, D, b);
#endif
}

inline void subtract_vectors(double *result_vec, const double *vec1,
                             const double *vec2, const int N,
                             const double scale = 1.0) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        result_vec[i] = vec1[i] - scale * vec2[i];
    }
}

inline void sum_vectors(double *result_vec, const double *vec1,
                        const double *vec2, const int N,
                        const double scale = 1.0) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        result_vec[i] = vec1[i] + scale * vec2[i];
    }
}

inline void elemwise_mult_vectors(double *result_vec, const double *vec1,
                                  const double *vec2, const int N,
                                  const double scale = 1.0) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        result_vec[i] = vec1[i] * scale * vec2[i];
    }
}

inline void elemwise_div_vectors(double *result_vec, const double *vec1,
                                 const double *vec2, const int N,
                                 const double scale = 1.0) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        result_vec[i] = vec1[i] / (scale * vec2[i]);
    }
}

inline void compute_residual(const MatrixCRS *A, const double *x,
                             const double *b, double *residual, double *tmp,
                             Interface *smax = nullptr,
                             const std::string kernel_name = "") {

    spmv(A, x, tmp SMAX_ARGS(0, smax, kernel_name));
    subtract_vectors(residual, b, tmp, A->n_cols);
}

inline double infty_vec_norm(const double *vec, const int N) {
    double max_abs = 0.0;
    double curr_abs;
    for (int i = 0; i < N; ++i) {
        // TODO:: Hmmm...
        // curr_abs = std::abs(static_cast<double>(vec[i]));
        curr_abs = (vec[i] >= 0) ? vec[i] : -1 * vec[i];
        if (curr_abs > max_abs) {
            max_abs = curr_abs;
        }
    }

    return max_abs;
}

inline double infty_mat_norm(const MatrixCRS *A) {
    double max_row_sum = 0.0;

#pragma omp parallel for reduction(max : max_row_sum) schedule(static)
    for (int row = 0; row < A->n_rows; ++row) {
        double row_sum = 0.0;
        for (int idx = A->row_ptr[row]; idx < A->row_ptr[row + 1]; ++idx) {
            row_sum += std::abs(A->val[idx]);
        }
        max_row_sum = std::max(max_row_sum, row_sum);
    }

    return max_row_sum;
}

inline double euclidean_vec_norm(const double *vec, int N) {
    double tmp = 0.0;

#pragma omp parallel for reduction(+ : tmp) schedule(static)
    for (int i = 0; i < N; ++i) {
        tmp += vec[i] * vec[i];
    }

    return std::sqrt(tmp);
}

inline double dot(const double *vec1, const double *vec2, const int N) {
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < N; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

inline void scale(double *result_vec, const double *vec, const double scalar,
                  const int N) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        result_vec[i] = vec[i] * scalar;
    }
}

inline void init_dense_identity_matrix(double *mat, const int n_rows,
                                       const int n_cols) {
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            if (i == j) {
                mat[n_cols * i + j] = 1.0;
            } else {
                mat[n_cols * i + j] = 0.0;
            }
        }
    }
}

inline void init_vector(double *vec, double val, long size) {
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        vec[i] = val;
    }
}

inline void copy_dense_matrix(double *new_mat, const double *old_mat,
                              const int n_rows, const int n_cols) {
    for (int row = 0; row < n_rows; ++row) {
        for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
            new_mat[n_cols * row + col_idx] = old_mat[n_cols * row + col_idx];
        }
    }
}

inline void copy_vector(double *output, const double *input, const int n_rows) {
#pragma omp parallel for
    for (int row = 0; row < n_rows; ++row) {
        output[row] = input[row];
    }
}

inline void dgemm_transpose1(double *A, double *B, double *C, int n_rows_A,
                             int n_cols_A, int n_cols_B) {
    for (int i = 0; i < n_rows_A; ++i) {
        for (int j = 0; j < n_cols_B; ++j) {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k) {
                tmp += A[k * n_rows_A + i] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

inline void dgemm_transpose2(double *A, double *B, double *C, int n_rows_A,
                             int n_cols_A, int n_cols_B) {
    for (int i = 0; i < n_rows_A; ++i) {
        for (int j = 0; j < n_cols_B; ++j) {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k) {
                tmp += A[i * n_cols_A + k] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

inline void dgemm(double *A, double *B, double *C, int n_rows_A, int n_cols_A,
                  int n_cols_B) {
    for (int i = 0; i < n_rows_A; ++i) {
        for (int j = 0; j < n_cols_B; ++j) {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k) {
                tmp += A[i * n_cols_A + k] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

inline void dgemv(const double *A, const double *x, double *y, int n_rows_A,
                  int n_cols_A, double alpha = 1.0
                  // double beta = 1.0
) {
    for (int i = 0; i < n_rows_A; ++i) {
        // y[i] *= beta;
        y[i] = 0.0;
        for (int j = 0; j < n_cols_A; ++j) {
            y[i] += alpha * A[i * n_cols_A + j] * x[j];
        }
    }
}

inline void two_stage_gauss_seidel(const MatrixCRS *strict, double *tmp,
                                   double *work, double *D_inv, double *input,
                                   double *output, const int N, int offset = 0,
                                   Interface *smax = nullptr,
                                   const std::string &kernel_name = "") {
    elemwise_mult_vectors(work, D_inv, input, N);

    copy_vector(output, work, N);

    for (int inner = 1; inner <= PRECOND_INNER_ITERS; ++inner) {

        spmv(strict, work, tmp SMAX_ARGS(0, smax, kernel_name));

        elemwise_mult_vectors(tmp, D_inv, tmp, N, -1.0);

        std::swap(work, tmp);
#ifdef USE_SMAX
        smax->kernel(kernel_name)->swap_operands();
#endif
        sum_vectors(output, output, work, N);
    }
}

// Computes z <- M^{-1}y
inline void apply_preconditioner(const PrecondType preconditioner, const int N,
                                 const MatrixCRS *L_strict,
                                 const MatrixCRS *U_strict, double *A_D,
                                 double *A_D_inv, double *L_D, double *U_D,
                                 double *output, double *input, double *tmp,
                                 double *work, int offset = 0,
                                 Interface *smax = nullptr,
                                 const std::string kernel_name = "") {

    IF_DEBUG_MODE_FINE(
        SanityChecker::print_vector(input, N, "before precond:"));

    // clang-format off
    for (int i = 0; i < PRECOND_OUTER_ITERS; ++i) {
        if (preconditioner == PrecondType::Jacobi) {
            elemwise_div_vectors(output, input, A_D, N);
        } else if (preconditioner == PrecondType::GaussSeidel) {
            sptrsv(L_strict, output, A_D, input SMAX_ARGS(0, smax, kernel_name));
        } else if (preconditioner == PrecondType::BackwardsGaussSeidel) {
            bsptrsv(U_strict, output, A_D, input SMAX_ARGS(0, smax, kernel_name));
        } else if (preconditioner == PrecondType::SymmetricGaussSeidel) {
            // tmp <- (L+D)^{-1}*r
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before lower solve"));
            sptrsv(L_strict, tmp, A_D, input SMAX_ARGS(0, smax, std::string(kernel_name + "_lower")));

            // tmp <- D(L+D)^{-1}*r
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before divide"));
            elemwise_mult_vectors(tmp, tmp, A_D, N);

            // z <- (L+U)^{-1}*tmp
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before upper solve"));
            bsptrsv(U_strict, output, A_D, tmp SMAX_ARGS(0, smax, std::string(kernel_name + "_upper")));
        
        } else if (preconditioner == PrecondType::TwoStageGS) {
            two_stage_gauss_seidel(L_strict, tmp, work, A_D_inv, input,
                            output, N SMAX_ARGS(0, smax, std::string(kernel_name)));
        } else if (preconditioner == PrecondType::SymmetricTwoStageGS) {
            two_stage_gauss_seidel(L_strict, tmp, work, A_D_inv, input,
                            output, N SMAX_ARGS(0, smax, std::string(kernel_name + "_lower")));

            elemwise_mult_vectors(output, output, A_D, N);

            two_stage_gauss_seidel(U_strict, tmp, work, A_D_inv, output,
                            output, N SMAX_ARGS(0, smax, std::string(kernel_name + "_upper")));
        } else if (preconditioner == PrecondType::ILU0 || preconditioner == PrecondType::ILUT) {
            // tmp <- L^{-1}*r
            // NOTE: L_D := ones(N), since we don't divide by anything
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before lower solve"));
            sptrsv(L_strict, tmp, L_D, input SMAX_ARGS(0, smax, std::string(kernel_name + "_lower")));

            // z <- U^{-1}*tmp
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before upper solve"));
            bsptrsv(U_strict, output, U_D, tmp SMAX_ARGS(0, smax, std::string(kernel_name + "_upper")));
        }
        else {
            // TODO: Would be great to think of a way around this
            copy_vector(output, input, N);
        }
    }

    // clang-format on
    IF_DEBUG_MODE_FINE(
        SanityChecker::print_vector(output, N, "after precond:"));
}
