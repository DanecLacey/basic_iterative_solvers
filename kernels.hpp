#ifndef KERNELS_HPP
#define KERNELS_HPP

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

inline void native_spmv(const MatrixCRS *crs_mat, const double *x, double *y) {
#pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("spmv");
#endif
#pragma omp for schedule(static)
        for (int row = 0; row < crs_mat->n_rows; ++row) {
            double tmp = 0.0;
#pragma omp simd reduction(+ : tmp)
            for (int nz_idx = crs_mat->row_ptr[row];
                 nz_idx < crs_mat->row_ptr[row + 1]; ++nz_idx) {
                tmp += crs_mat->val[nz_idx] * x[crs_mat->col[nz_idx]];
            }
            y[row] = tmp;
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("spmv");
#endif
    }
}

inline void spmv(const MatrixCRS *crs_mat, const double *x, double *y,
                 int offset = 0, Interface *smax = nullptr,
                 const std::string kernel_name = "") {
#if USE_SMAX
    smax->kernel(kernel_name.c_str())->run(0, offset, 0);
#else
    native_spmv(crs_mat, x, y);
#endif
}

inline void native_sptrsv(const MatrixCRS *crs_mat_L, double *x,
                          const double *D, const double *b) {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("sptrsv");
#endif

    double sum;
    for (int row = 0; row < crs_mat_L->n_rows; ++row) {
        sum = 0.0;
        int row_start = crs_mat_L->row_ptr[row];
        int row_stop = crs_mat_L->row_ptr[row + 1];

        for (int nz_idx = row_start; nz_idx < row_stop; ++nz_idx) {
            sum += crs_mat_L->val[nz_idx] * x[crs_mat_L->col[nz_idx]];
        }

        x[row] = (b[row] - sum) / D[row];
    }

#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("sptrsv");
#endif
}

inline void sptrsv(const MatrixCRS *crs_mat_L, double *x, const double *D,
                   const double *b, int offset = 0, Interface *smax = nullptr,
                   const std::string kernel_name = "") {
#ifdef USE_SMAX
    smax->kernel(kernel_name.c_str())->run(0, offset, 0);
#else
    native_sptrsv(crs_mat_L, x, D, b);
#endif
}

inline void native_bsptrsv(const MatrixCRS *crs_mat_U, double *x,
                           const double *D, const double *b) {
#ifdef USE_LIKWID
    LIKWID_MARKER_START("backwards-sptrsv");
#endif
    for (int row = crs_mat_U->n_rows - 1; row >= 0; --row) {
        double sum = 0.0;
        int row_start = crs_mat_U->row_ptr[row];
        int row_stop = crs_mat_U->row_ptr[row + 1];

        for (int nz_idx = row_start; nz_idx < row_stop; ++nz_idx) {
            sum += crs_mat_U->val[nz_idx] * x[crs_mat_U->col[nz_idx]];
        }

        x[row] = (b[row] - sum) / D[row];
    }
#ifdef USE_LIKWID
    LIKWID_MARKER_STOP("backwards-sptrsv");
#endif
}

inline void bsptrsv(const MatrixCRS *crs_mat_U, double *x, const double *D,
                    const double *b, int offset = 0, Interface *smax = nullptr,
                    const std::string kernel_name = "") {
#if USE_SMAX
    smax->kernel(kernel_name.c_str())->run(0, offset, 0);
#else
    native_bsptrsv(crs_mat_U, x, D, b);
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

inline void compute_residual(const MatrixCRS *crs_mat, const double *x,
                             const double *b, double *residual, double *tmp,
                             Interface *smax = nullptr,
                             const std::string kernel_name = "") {

    spmv(crs_mat, x, tmp SMAX_ARGS(0, smax, kernel_name));
    subtract_vectors(residual, b, tmp, crs_mat->n_cols);
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

inline double infty_mat_norm(const MatrixCRS *crs_mat) {
    double max_row_sum = 0.0;

#pragma omp parallel for reduction(max : max_row_sum) schedule(static)
    for (int row = 0; row < crs_mat->n_rows; ++row) {
        double row_sum = 0.0;
        for (int idx = crs_mat->row_ptr[row]; idx < crs_mat->row_ptr[row + 1];
             ++idx) {
            row_sum += std::abs(crs_mat->val[idx]);
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

// clang-format off
// New function for forward substitution with a unit-diagonal L matrix.
inline void sptrsv_unit_diag(const MatrixCRS *crs_mat_L_strict, double *x, const double *b) {
    // Solves Lx = b where L is unit lower triangular.
    for (int row = 0; row < crs_mat_L_strict->n_rows; ++row) {
        double sum = 0.0;
        for (int nz_idx = crs_mat_L_strict->row_ptr[row]; nz_idx < crs_mat_L_strict->row_ptr[row + 1]; ++nz_idx) {
            sum += crs_mat_L_strict->val[nz_idx] * x[crs_mat_L_strict->col[nz_idx]];
        }
        // Diagonal is 1.0, so no division needed.
        x[row] = b[row] - sum;
    }
}
// clang-format on

// Computes z <- M^{-1}y
inline void apply_preconditioner(const PrecondType preconditioner, const int N,
                                 const MatrixCRS *crs_mat_L_strict,
                                 const MatrixCRS *crs_mat_U_strict,
                                 const MatrixCRS *L_factor,
                                 const MatrixCRS *U_factor, double *D,
                                 double *D_inv, double *output, double *input,
                                 double *tmp, double *work, int offset = 0,
                                 Interface *smax = nullptr,
                                 const std::string kernel_name = "") {

    IF_DEBUG_MODE_FINE(
        SanityChecker::print_vector(input, N, "before precond:"));

    // clang-format off
    for (int i = 0; i < PRECOND_OUTER_ITERS; ++i) {
        if (preconditioner == PrecondType::Jacobi) {
            elemwise_div_vectors(output, input, D, N);
        } else if (preconditioner == PrecondType::GaussSeidel) {
            sptrsv(crs_mat_L_strict, output, D, input SMAX_ARGS(0, smax, kernel_name));
        } else if (preconditioner == PrecondType::BackwardsGaussSeidel) {
            bsptrsv(crs_mat_U_strict, output, D, input SMAX_ARGS(0, smax, kernel_name));
        } else if (preconditioner == PrecondType::SymmetricGaussSeidel) {
            // tmp <- (L+D)^{-1}*r
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before lower solve"));
            sptrsv(crs_mat_L_strict, tmp, D, input SMAX_ARGS(0, smax, std::string(kernel_name + "_lower")));

            // tmp <- D(L+D)^{-1}*r
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before divide"));
            elemwise_mult_vectors(tmp, tmp, D, N);

            // z <- (L+U)^{-1}*tmp
            IF_DEBUG_MODE_FINE(SanityChecker::print_vector(tmp, N, "tmp before upper solve"));
            bsptrsv(crs_mat_U_strict, output, D, tmp SMAX_ARGS(0, smax, std::string(kernel_name + "_upper")));
        
        } else if (preconditioner == PrecondType::TwoStageGS) {

            elemwise_mult_vectors(work, D_inv, input, N);

            for (int inner = 1; inner <= PRECOND_INNER_ITERS; ++inner) {
                // tmp = L * work
                spmv(crs_mat_L_strict, work, tmp SMAX_ARGS(0, smax, std::string("precon_spmv")));
#ifdef USE_SMAX
                smax->kernel("precon_spmv")->swap_operands();
#endif
                elemwise_mult_vectors(tmp, D_inv, tmp, N, -1.0);

                sum_vectors(work, work, tmp, N);
            }

            copy_vector(output, work, N);

        } else if (preconditioner == PrecondType::ILU0) {
            // Copied from Aashutosh branch //
            
            // ILU(0) solve: Mz = r  =>  (LU)z = r
            // We use `tmp` as the intermediate vector y.

            // Step 1: Solve Ly = r for y. Store result in `tmp`.
            // L_factor is strictly lower, so we need a unit-diagonal solve.
            sptrsv_unit_diag(L_factor, tmp, input);
            
            // Step 2: Solve Uz = y for z. Store final result in `vec`.
            // The rhs for this solve is the `tmp` vector we just computed.
            bsptrsv(U_factor, output, D, tmp);
        } else {
            // TODO: Would be great to think of a way around this
            copy_vector(output, input, N);
        }
    }

    // clang-format on
    IF_DEBUG_MODE_FINE(
        SanityChecker::print_vector(output, N, "after precond:"));
}

#endif
