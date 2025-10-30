#pragma once

#include "../common.hpp"
#include "../sparse_matrix.hpp"

inline void split_LU_old(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                         MatrixCRS *U, MatrixCRS *U_strict) {
    int D_nz_count = 0;

    // These will safely accumulate the counts in parallel.
    int l_nnz = 0;
    int l_strict_nnz = 0;
    int u_nnz = 0;
    int u_strict_nnz = 0;

    // Set fixed-size metadata.
    U->n_rows = A->n_rows;
    U->n_cols = A->n_cols;
    U_strict->n_rows = A->n_rows;
    U_strict->n_cols = A->n_cols;
    L->n_rows = A->n_rows;
    L->n_cols = A->n_cols;
    L_strict->n_rows = A->n_rows;
    L_strict->n_cols = A->n_cols;

// We use local variables in the OpenMP loop.
// The loop iterates over rows and increments the local counters.
#pragma omp parallel for reduction(+ : l_nnz, l_strict_nnz, u_nnz,             \
                                       u_strict_nnz) schedule(static)
    for (int i = 0; i < A->n_rows; ++i) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A->col[idx];

            if (col <= i) {
                l_nnz++; // Increment local variable
                if (col < i) {
                    l_strict_nnz++; // Increment local variable
                }
            }
            if (col >= i) {
                u_nnz++; // Increment local variable
                if (col > i) {
                    u_strict_nnz++; // Increment local variable
                }
            }
        }
    }

    // Now, we assign the final, correct counts to the struct members
    // This happens *after* the parallel region is finished.
    L->nnz = l_nnz;
    L_strict->nnz = l_strict_nnz;
    U->nnz = u_nnz;
    U_strict->nnz = u_strict_nnz;

    // Allocate heap space and assign known metadata
    L->col = new int[L->nnz];
    L->row_ptr = new int[A->n_rows + 1];
    L->val = new double[L->nnz];
    L->row_ptr[0] = 0;

    L_strict->col = new int[L_strict->nnz];
    L_strict->row_ptr = new int[A->n_rows + 1];
    L_strict->val = new double[L_strict->nnz];
    L_strict->row_ptr[0] = 0;

    U->col = new int[U->nnz];
    U->row_ptr = new int[A->n_rows + 1];
    U->val = new double[U->nnz];
    U->row_ptr[0] = 0;

    U_strict->col = new int[U_strict->nnz];
    U_strict->row_ptr = new int[A->n_rows + 1];
    U_strict->val = new double[U_strict->nnz];
    U_strict->row_ptr[0] = 0;

    // Assign nonzeros
    int L_count = 0;
    int L_strict_count = 0;
    int U_count = 0;
    int U_strict_count = 0;
    for (int i = 0; i < A->n_rows; ++i) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A->col[idx];
            double val = A->val[idx];

            if (col <= i) {
                L->col[L_count] = col;
                L->val[L_count++] = val;
                if (col < i) {
                    L_strict->col[L_strict_count] = col;
                    L_strict->val[L_strict_count++] = val;
                }
            }
            if (col >= i) {
                U->col[U_count] = col;
                U->val[U_count++] = val;
                if (col > i) {
                    U_strict->col[U_strict_count] = col;
                    U_strict->val[U_strict_count++] = val;
                }
            }
        }

        // Update row pointers
        L->row_ptr[i + 1] = L_count;
        L_strict->row_ptr[i + 1] = L_strict_count;
        U->row_ptr[i + 1] = U_count;
        U_strict->row_ptr[i + 1] = U_strict_count;
    }
}

// Based on "extract_D_L_U" in SmaxKernels
inline void split_LU_new(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                         MatrixCRS *U, MatrixCRS *U_strict) {

    // Clear data from targets
    // NOTE: Why not just use "clear()" methods?
    // clang-format off
    if (L->row_ptr != nullptr) { delete[] L->row_ptr; L->row_ptr = nullptr; }
    if (L->col != nullptr)     { delete[] L->col;     L->col = nullptr; }
    if (L->val != nullptr)     { delete[] L->val;     L->val = nullptr; }

    if (L_strict->row_ptr != nullptr) { delete[] L_strict->row_ptr; L_strict->row_ptr = nullptr; }
    if (L_strict->col != nullptr)     { delete[] L_strict->col;     L_strict->col = nullptr; }
    if (L_strict->val != nullptr)     { delete[] L_strict->val;     L_strict->val = nullptr; }

    if (U->row_ptr != nullptr) { delete[] U->row_ptr; U->row_ptr = nullptr; }
    if (U->col != nullptr)     { delete[] U->col;     U->col = nullptr; }
    if (U->val != nullptr)     { delete[] U->val;     U->val = nullptr; }

    if (U_strict->row_ptr != nullptr) { delete[] U_strict->row_ptr; U_strict->row_ptr = nullptr; }
    if (U_strict->col != nullptr)     { delete[] U_strict->col;     U_strict->col = nullptr; }
    if (U_strict->val != nullptr)     { delete[] U_strict->val;     U_strict->val = nullptr; }

    // Make nnz per row storages to simultaneously obtain nnz and row ptr
    auto L_nnz_row = new int[A->n_rows];
    auto L_strict_nnz_row = new int[A->n_rows];
    auto U_nnz_row = new int[A->n_rows];
    auto U_strict_nnz_row = new int[A->n_rows];
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A->n_rows; i++) {
        L_nnz_row[i] = 0;
        L_strict_nnz_row[i] = 0;
        U_nnz_row[i] = 0;
        U_strict_nnz_row[i] = 0;
    }

    // clang-format on
    L->nnz = L_strict->nnz = U->nnz = U_strict->nnz = 0;

    // Count nnz
    int L_tmp_nnz = 0;
    int L_strict_tmp_nnz = 0;
    int U_tmp_nnz = 0;
    int U_strict_tmp_nnz = 0;

#pragma omp parallel for reduction(+ : L_tmp_nnz, L_strict_tmp_nnz, U_tmp_nnz, \
                                       U_strict_tmp_nnz) schedule(static)
    for (int i = 0; i < A->n_rows; ++i) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A->col[idx];

            if (col < i) {
                L_nnz_row[i] += 1;
                L_strict_nnz_row[i] += 1;
                ++L_tmp_nnz;
                ++L_strict_tmp_nnz;
            }
            if (col == i) {
                U_nnz_row[i] += 1;
                L_nnz_row[i] += 1;
                ++L_tmp_nnz;
                ++U_tmp_nnz;
            }
            if (col > i) {
                U_nnz_row[i] += 1;
                U_strict_nnz_row[i] += 1;
                ++U_tmp_nnz;
                ++U_strict_tmp_nnz;
            }
        }
    }

    // Make tmp structs
    int N = A->n_rows;
    MatrixCRS *L_tmp = new MatrixCRS(N, N, L_tmp_nnz);
    L_tmp->row_ptr[0] = 0;
    MatrixCRS *L_strict_tmp = new MatrixCRS(N, N, L_strict_tmp_nnz);
    L_strict_tmp->row_ptr[0] = 0;
    MatrixCRS *U_tmp = new MatrixCRS(N, N, U_tmp_nnz);
    U_tmp->row_ptr[0] = 0;
    MatrixCRS *U_strict_tmp = new MatrixCRS(N, N, U_strict_tmp_nnz);
    U_strict_tmp->row_ptr[0] = 0;

    // Compute actual row_ptr from nnz per row
    for (int i = 1; i <= A->n_rows; i++) {
        L_tmp->row_ptr[i] = L_tmp->row_ptr[i - 1] + L_nnz_row[i - 1];
        L_strict_tmp->row_ptr[i] =
            L_strict_tmp->row_ptr[i - 1] + L_strict_nnz_row[i - 1];
        U_tmp->row_ptr[i] = U_tmp->row_ptr[i - 1] + U_nnz_row[i - 1];
        U_strict_tmp->row_ptr[i] =
            U_strict_tmp->row_ptr[i - 1] + U_strict_nnz_row[i - 1];
    }

    // Assign nonzeros
    int L_tmp_count = 0;
    int L_strict_tmp_count = 0;
    int U_tmp_count = 0;
    int U_strict_tmp_count = 0;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A->n_rows; ++i) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];

        int L_tmp_row_start = L_tmp->row_ptr[i];

        int L_strict_tmp_row_start = L_strict_tmp->row_ptr[i];

        int U_tmp_row_start = U_tmp->row_ptr[i];

        int U_strict_tmp_row_start = U_strict_tmp->row_ptr[i];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A->col[idx];
            double val = A->val[idx];

            if (col < i) {
                L_tmp->val[L_tmp_row_start] = val;
                L_tmp->col[L_tmp_row_start++] = col;
                L_strict_tmp->val[L_strict_tmp_row_start] = val;
                L_strict_tmp->col[L_strict_tmp_row_start++] = col;
            }
            if (col == i) {
                L_tmp->val[L_tmp_row_start] = val;
                L_tmp->col[L_tmp_row_start++] = col;
                U_tmp->val[U_tmp_row_start] = val;
                U_tmp->col[U_tmp_row_start++] = col;
            }
            if (col > i) {
                U_tmp->val[U_tmp_row_start] = val;
                U_tmp->col[U_tmp_row_start++] = col;
                U_strict_tmp->val[U_strict_tmp_row_start] = val;
                U_strict_tmp->col[U_strict_tmp_row_start++] = col;
            }
        }
    }

    // Give L and U metadata
    L->n_rows = A->n_rows;
    L->n_cols = A->n_cols;
    L->nnz = L_tmp->nnz;
    L->val = new double[L_tmp->nnz];
    L->col = new int[L_tmp->nnz];
    L->row_ptr = new int[L->n_rows + 1];
    L->row_ptr[0] = 0;

    L_strict->n_rows = A->n_rows;
    L_strict->n_cols = A->n_cols;
    L_strict->nnz = L_strict_tmp->nnz;
    L_strict->val = new double[L_strict_tmp->nnz];
    L_strict->col = new int[L_strict_tmp->nnz];
    L_strict->row_ptr = new int[L_strict->n_rows + 1];
    L_strict->row_ptr[0] = 0;

    U->n_rows = A->n_rows;
    U->n_cols = A->n_cols;
    U->nnz = U_tmp->nnz;
    U->val = new double[U_tmp->nnz];
    U->col = new int[U_tmp->nnz];
    U->row_ptr = new int[U->n_rows + 1];
    U->row_ptr[0] = 0;

    U_strict->n_rows = A->n_rows;
    U_strict->n_cols = A->n_cols;
    U_strict->nnz = U_strict_tmp->nnz;
    U_strict->val = new double[U_strict_tmp->nnz];
    U_strict->col = new int[U_strict_tmp->nnz];
    U_strict->row_ptr = new int[U_strict->n_rows + 1];
    U_strict->row_ptr[0] = 0;

    // Finally, numa-friendly copy tmp matrices to L and U
    *L = *L_tmp;
    *L_strict = *L_strict_tmp;
    *U = *U_tmp;
    *U_strict = *U_strict_tmp;

    delete L_tmp;
    delete L_strict_tmp;
    delete U_tmp;
    delete U_strict_tmp;
    delete[] L_nnz_row;
    delete[] L_strict_nnz_row;
    delete[] U_nnz_row;
    delete[] U_strict_nnz_row;
}

inline void split_LU(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                     MatrixCRS *U, MatrixCRS *U_strict) {
#if 0
    split_LU_old(A, L, L_strict, U, U_strict);
#elif 1
    split_LU_new(A, L, L_strict, U, U_strict);
#endif
}

inline void factor_ILU0_old(Timers *timers, const MatrixCRS *A, MatrixCRS *L,
                            MatrixCRS *L_strict, double *L_D, MatrixCRS *U,
                            MatrixCRS *U_strict, double *U_D,
                            Interface *smax = nullptr) {
    int n = A->n_rows;

    // These will store the final factors, built row-by-row.
    std::vector<std::vector<std::pair<int, double>>> L_rows(n);
    std::vector<std::vector<std::pair<int, double>>> U_rows(n);

    // Workspace for the current row i. `w_vals` holds the values,
    // and `w_indices` tracks the non-zero pattern of the original row A(i, *).
    std::vector<double> w_vals(n, 0.0);
    std::vector<int> w_indices;
    w_indices.reserve(A->nnz / n + 10); // Pre-allocate based on average nnz/row

    // Main loop for each row `i` of the factorization.
    for (int i = 0; i < n; ++i) {

        // --- Step 1: Scatter row A(i,*) into the workspace ---
        timers->preprocessing_factor_1_time->start();
        // This establishes the fixed sparsity pattern for the ILU(0)
        // calculations.
        for (int j_pos = A->row_ptr[i]; j_pos < A->row_ptr[i + 1]; ++j_pos) {
            int j = A->col[j_pos];
            w_vals[j] = A->val[j_pos]; // Scattered CRS -> dense
            w_indices.push_back(j);    // Packed
        }
        // Sort indices to process dependencies (k) in increasing order.
        std::sort(w_indices.begin(), w_indices.end());
        timers->preprocessing_factor_1_time->stop();

        // --- Step 2: Elimination loop ---
        timers->preprocessing_factor_2_time->start();
        // For each non-zero k < i in the current row's pattern...
        for (int k : w_indices) {
            if (k >= i)
                break;

            // Find the pivot U(k,k) from the previously computed k-th row of U.
            double pivot = 0.0;
            for (const auto &u_entry : U_rows[k]) {
                if (u_entry.first == k) {
                    pivot = u_entry.second;
                    break;
                }
            }

            // Check for unstable pivot. If it's bad, we can't use this row for
            // elimination.
            if (std::abs(pivot) < 1e-16)
                continue;

            // Compute the L-factor L(i,k)
            double factor = w_vals[k] / pivot;
            w_vals[k] = factor; // Store it in the workspace.

            // Perform sparse update: w(j) -= L(i,k) * U(k,j)
            // We only update elements j that are already in the sparsity
            // pattern of row i.
            for (const auto &u_entry : U_rows[k]) {
                int j = u_entry.first;
                // Check if an element A(i,j) exists. w_vals[j] != 0 is a proxy
                // for this.
                if (j > k && w_vals[j] != 0.0) {
                    w_vals[j] -= factor * u_entry.second;
                }
            }
        } // End of elimination for row i
        timers->preprocessing_factor_2_time->stop();

        // --- Step 3: Gather the computed row into final L and U structures ---
        timers->preprocessing_factor_3_time->start();
        timers->preprocessing_factor_3_1_time->start();
        std::vector<std::pair<int, double>> l_row_final, u_row_final;
        double u_diag = 0.0;

        for (int j : w_indices) {
            if (j < i) {
                l_row_final.push_back({j, w_vals[j]});
            } else if (j == i) {
                u_diag = w_vals[j];
            } else { // j > i
                u_row_final.push_back({j, w_vals[j]});
            }
        }
        timers->preprocessing_factor_3_1_time->stop();
        timers->preprocessing_factor_3_2_time->start();

        // Robust pivot handling for the diagonal element U(i,i)
        if (std::abs(u_diag) < ILU0_PIVOT_TOLERANCE) {
            u_diag = (u_diag >= 0 ? 1.0 : -1.0) * ILU0_PIVOT_REPLACEMENT;
        }
        u_row_final.push_back({i, u_diag});
        timers->preprocessing_factor_3_2_time->stop();
        timers->preprocessing_factor_3_3_time->start();
        std::sort(u_row_final.begin(), u_row_final.end());
        timers->preprocessing_factor_3_3_time->stop();
        timers->preprocessing_factor_3_4_time->start();

        L_rows[i] = l_row_final;
        U_rows[i] = u_row_final;

        // --- Step 4: Cleanup workspace for the next iteration ---
        for (int j : w_indices) {
            w_vals[j] = 0.0;
        }
        w_indices.clear();
        timers->preprocessing_factor_3_4_time->stop();
        timers->preprocessing_factor_3_time->stop();
    }

    // --- Finalization: Convert vector-of-vectors to CRS format ---
    timers->preprocessing_factor_4_time->start();
    // Count strict nnz and how many diagonals are missing
    int l_nnz_strict = 0;
    int missing_diags = 0;
    int u_nnz = 0;
#pragma omp parallel for reduction(+ : l_nnz_strict, u_nnz, missing_diags)
    for (int i = 0; i < n; ++i) {
        l_nnz_strict += (int)L_rows[i].size();
        u_nnz += U_rows[i].size();
        bool has_diag = false;
        for (const auto &e : L_rows[i]) {
            if (e.first == i) {
                has_diag = true;
                break;
            }
        }
        if (!has_diag)
            ++missing_diags;
    }
    timers->preprocessing_factor_4_time->stop();
    timers->preprocessing_factor_5_time->start();

    int l_nnz_with_diag = l_nnz_strict + missing_diags;

    // Allocate all required matrices
    L_strict->n_rows = L->n_rows = n;
    L_strict->n_cols = L->n_cols = n;
    L_strict->nnz = l_nnz_strict;
    L->nnz = l_nnz_with_diag;
    L->row_ptr = new int[n + 1];
    L->col = new int[L->nnz];
    L->val = new double[L->nnz];
    L_strict->row_ptr = new int[n + 1];
    L_strict->col = new int[L_strict->nnz];
    L_strict->val = new double[L_strict->nnz];

    U->n_rows = n;
    U->n_cols = n;
    U->nnz = u_nnz;
    U->row_ptr = new int[n + 1];
    U->col = new int[u_nnz];
    U->val = new double[u_nnz];

    // Fill L (append diag if missing) and L_strict (strict only)
    int posL = 0;
    int posLs = 0;
    for (int i = 0; i < n; ++i) {
        // L (with diag)
        L->row_ptr[i] = posL;
        bool has_diag = false;
        for (const auto &entry : L_rows[i]) {
            L->col[posL] = entry.first;
            L->val[posL] = entry.second;
            if (entry.first == i)
                has_diag = true;
            ++posL;
        }
        if (!has_diag) {
            L->col[posL] = i;
            L->val[posL] = 1.0;
            ++posL;
        }

        // L_strict (copy strict-lower entries only)
        L_strict->row_ptr[i] = posLs;
        for (const auto &entry : L_rows[i]) {
            // copy every entry, including a diagonal if it was present in
            // L_rows: but the "strict" requirement in your earlier text
            // suggested L_rows is already strictly lower. If you truly want to
            // exclude diag entries from L_strict, skip entry.first == i here.
            if (entry.first != i) {
                L_strict->col[posLs] = entry.first;
                L_strict->val[posLs] = entry.second;
                ++posLs;
            }
        }
    }
    timers->preprocessing_factor_5_time->stop();
    timers->preprocessing_factor_6_time->start();
    L->row_ptr[n] = posL;
    L_strict->row_ptr[n] = posLs;

    // For a unit-diagonal L-solve, L_D must be all ones.
    for (int i = 0; i < n; ++i)
        L_D[i] = 1.0;

    // Populate U and peel its diagonal into U_D
    int pos = 0;

    for (int i = 0; i < n; ++i) {
        U->row_ptr[i] = pos;
        for (const auto &entry : U_rows[i]) {
            U->col[pos] = entry.first;
            U->val[pos] = entry.second;
            if (entry.first == i) {
                U_D[i] = entry.second;
            }
            pos++;
        }
    }
    U->row_ptr[n] = pos;
    timers->preprocessing_factor_6_time->stop();

    // This isn't strictly needed if U_strict isn't used elsewhere, but good for
    // completeness
    split_LU(U, new MatrixCRS(), new MatrixCRS(), new MatrixCRS(), U_strict);
}

inline void factor_ILU0_new(Timers *timers, const MatrixCRS *A, MatrixCRS *L,
                            MatrixCRS *L_strict, double *L_D, MatrixCRS *U,
                            MatrixCRS *U_strict, double *U_D,
                            Interface *smax = nullptr) {
#ifdef USE_SMAX
    int n = A->n_rows;

    // These will store the final factors, built row-by-row.
    std::vector<std::vector<std::pair<int, double>>> L_rows(n);
    std::vector<std::vector<std::pair<int, double>>> U_rows(n);

    // Workspace for the current row i. `w_vals` holds the values,
    // and `w_indices` tracks the non-zero pattern of the original row A(i, *).
    std::vector<double> w_vals(n, 0.0);
    std::vector<int> w_indices;
    w_indices.reserve(A->nnz / n + 10); // Pre-allocate based on average nnz/row

    // Main loop for each row `i` of the factorization.
    int n_levels = smax->get_n_levels();
    for (int level = 0; level < n_levels; ++level) {
        int lvl_start = smax->get_level_ptr_at(level);
        int lvl_end = smax->get_level_ptr_at(level + 1);
#pragma omp parallel
        {
            std::vector<double> w_vals_private(n, 0.0);
            std::vector<int> w_indices_private;
            w_indices_private.reserve(A->nnz / n + 10);
#pragma omp for
            for (int i = lvl_start; i < lvl_end; ++i) {
                auto &w_vals = w_vals_private;
                auto &w_indices = w_indices_private;
                // --- Step 1: Scatter row A(i,*) into the workspace ---
                // This establishes the fixed sparsity pattern for the ILU(0)
                // calculations.
                for (int j_pos = A->row_ptr[i]; j_pos < A->row_ptr[i + 1];
                     ++j_pos) {
                    int j = A->col[j_pos];
                    w_vals[j] = A->val[j_pos]; // Scattered CRS -> dense
                    w_indices.push_back(j);    // Packed
                }
                // Sort indices to process dependencies (k) in increasing order.
                std::sort(w_indices.begin(), w_indices.end());

                // --- Step 2: Elimination loop ---
                // For each non-zero k < i in the current row's pattern...
                for (int k : w_indices) {
                    if (k >= i)
                        break;

                    // Find the pivot U(k,k) from the previously computed k-th
                    // row of U.
                    double pivot = 0.0;
                    for (const auto &u_entry : U_rows[k]) {
                        if (u_entry.first == k) {
                            pivot = u_entry.second;
                            break;
                        }
                    }

                    // Check for unstable pivot. If it's bad, we can't use this
                    // row for elimination.
                    if (std::abs(pivot) < 1e-16)
                        continue;

                    // Compute the L-factor L(i,k)
                    double factor = w_vals[k] / pivot;
                    w_vals[k] = factor; // Store it in the workspace.

                    // Perform sparse update: w(j) -= L(i,k) * U(k,j)
                    // We only update elements j that are already in the
                    // sparsity pattern of row i.
                    for (const auto &u_entry : U_rows[k]) {
                        int j = u_entry.first;
                        // Check if an element A(i,j) exists. w_vals[j] != 0 is
                        // a proxy for this.
                        if (j > k && w_vals[j] != 0.0) {
                            w_vals[j] -= factor * u_entry.second;
                        }
                    }
                } // End of elimination for row i

                // --- Step 3: Gather the computed row into final L and U
                // structures
                // ---
                std::vector<std::pair<int, double>> l_row_final, u_row_final;
                double u_diag = 0.0;

                for (int j : w_indices) {
                    if (j < i) {
                        l_row_final.push_back({j, w_vals[j]});
                    } else if (j == i) {
                        u_diag = w_vals[j];
                    } else { // j > i
                        u_row_final.push_back({j, w_vals[j]});
                    }
                }

                // Robust pivot handling for the diagonal element U(i,i)
                if (std::abs(u_diag) < ILU0_PIVOT_TOLERANCE) {
                    u_diag =
                        (u_diag >= 0 ? 1.0 : -1.0) * ILU0_PIVOT_REPLACEMENT;
                }
                u_row_final.push_back({i, u_diag});
                std::sort(u_row_final.begin(), u_row_final.end());

                L_rows[i] = l_row_final;
                U_rows[i] = u_row_final;

                // --- Step 4: Cleanup workspace for the next iteration ---
                for (int j : w_indices) {
                    w_vals[j] = 0.0;
                }
                w_indices.clear();
            }
        }
    }

    // --- Finalization: Convert vector-of-vectors to CRS format ---
    timers->preprocessing_factor_4_time->start();
    // Count strict nnz and how many diagonals are missing
    int l_nnz_strict = 0;
    int missing_diags = 0;
    int u_nnz = 0;
#pragma omp parallel for reduction(+ : l_nnz_strict, u_nnz, missing_diags)
    for (int i = 0; i < n; ++i) {
        l_nnz_strict += (int)L_rows[i].size();
        u_nnz += U_rows[i].size();
        bool has_diag = false;
        for (const auto &e : L_rows[i]) {
            if (e.first == i) {
                has_diag = true;
                break;
            }
        }
        if (!has_diag)
            ++missing_diags;
    }
    timers->preprocessing_factor_4_time->stop();
    timers->preprocessing_factor_5_time->start();

    int l_nnz_with_diag = l_nnz_strict + missing_diags;

    // Allocate all required matrices
    L_strict->n_rows = L->n_rows = n;
    L_strict->n_cols = L->n_cols = n;
    L_strict->nnz = l_nnz_strict;
    L->nnz = l_nnz_with_diag;
    L->row_ptr = new int[n + 1];
    L->col = new int[L->nnz];
    L->val = new double[L->nnz];
    L_strict->row_ptr = new int[n + 1];
    L_strict->col = new int[L_strict->nnz];
    L_strict->val = new double[L_strict->nnz];

    U->n_rows = n;
    U->n_cols = n;
    U->nnz = u_nnz;
    U->row_ptr = new int[n + 1];
    U->col = new int[u_nnz];
    U->val = new double[u_nnz];

    // Fill L (append diag if missing) and L_strict (strict only)
    int posL = 0;
    int posLs = 0;
    for (int i = 0; i < n; ++i) {
        // L (with diag)
        L->row_ptr[i] = posL;
        bool has_diag = false;
        for (const auto &entry : L_rows[i]) {
            L->col[posL] = entry.first;
            L->val[posL] = entry.second;
            if (entry.first == i)
                has_diag = true;
            ++posL;
        }
        if (!has_diag) {
            L->col[posL] = i;
            L->val[posL] = 1.0;
            ++posL;
        }

        // L_strict (copy strict-lower entries only)
        L_strict->row_ptr[i] = posLs;
        for (const auto &entry : L_rows[i]) {
            // copy every entry, including a diagonal if it was present in
            // L_rows: but the "strict" requirement in your earlier text
            // suggested L_rows is already strictly lower. If you truly want to
            // exclude diag entries from L_strict, skip entry.first == i here.
            if (entry.first != i) {
                L_strict->col[posLs] = entry.first;
                L_strict->val[posLs] = entry.second;
                ++posLs;
            }
        }
    }
    timers->preprocessing_factor_5_time->stop();
    timers->preprocessing_factor_6_time->start();
    L->row_ptr[n] = posL;
    L_strict->row_ptr[n] = posLs;

    // For a unit-diagonal L-solve, L_D must be all ones.
    for (int i = 0; i < n; ++i)
        L_D[i] = 1.0;

    // Populate U and peel its diagonal into U_D
    int pos = 0;

    for (int i = 0; i < n; ++i) {
        U->row_ptr[i] = pos;
        for (const auto &entry : U_rows[i]) {
            U->col[pos] = entry.first;
            U->val[pos] = entry.second;
            if (entry.first == i) {
                U_D[i] = entry.second;
            }
            pos++;
        }
    }
    U->row_ptr[n] = pos;
    timers->preprocessing_factor_6_time->stop();

    // This isn't strictly needed if U_strict isn't used elsewhere, but good for
    // completeness
    split_LU(U, new MatrixCRS(), new MatrixCRS(), new MatrixCRS(), U_strict);
#else
    printf("ERROR: factor_ILU0_new required SMAX library.\n");
#endif
}

inline void factor_ILU0(Timers *timers, const MatrixCRS *A, MatrixCRS *L,
                        MatrixCRS *L_strict, double *L_D, MatrixCRS *U,
                        MatrixCRS *U_strict, double *U_D,
                        Interface *smax = nullptr) {
#if 1
    factor_ILU0_new(timers, A, L, L_strict, L_D, U, U_strict,
                    U_D SMAX_ARGS(smax));
#elif 0
    factor_ILU0_old(timers, A, L, L_strict, L_D, U, U_strict,
                    U_D SMAX_ARGS(smax));
#endif
}

inline void peel_diag_crs_old(MatrixCRS *A, double *D,
                              double *D_inv = nullptr) {
    for (int row_idx = 0; row_idx < A->n_rows; ++row_idx) {
        int row_start = A->row_ptr[row_idx];
        int row_end = A->row_ptr[row_idx + 1] -
                      1; // Index of the last element in the current row
        int diag_j = -1; // Initialize diag_j to -1 (indicating diagonal not
                         // found yet)

        // Find the diagonal element in this row (since rows in CRS need not
        // be column-sorted)
        for (int j = row_start; j <= row_end; ++j) {
            if (A->col[j] == row_idx) {
                diag_j = j; // Store the index of the diagonal element
                D[row_idx] = A->val[j]; // Extract the diagonal value

                // Check if the diagonal value is very close to zero
                if (std::abs(D[row_idx]) < 1e-16) {
                    SanityChecker::zero_diag(
                        row_idx); // Call sanity checker for zero diagonal
                }

                if (D_inv)
                    D_inv[row_idx] = 1.0 / D[row_idx];
            }
        }

        // If no diagonal element was found for this row
        if (diag_j < 0) {
            SanityChecker::no_diag(
                row_idx); // Call sanity checker for missing diagonal
        }

        // If a diagonal element was found AND it's not already at the end
        // of the row's non-zeros, swap it into the last slot of the current
        // row's non-zero entries.
        if (diag_j >= 0 &&
            diag_j != row_end) { // Ensure diag_j is valid before swapping
            std::swap(A->col[diag_j], A->col[row_end]);
            std::swap(A->val[diag_j], A->val[row_end]);
        }
    }
}

inline void peel_diag_crs_new(MatrixCRS *A, double *D,
                              double *D_inv = nullptr) {
#pragma omp parallel for schedule(static)
    for (int row_idx = 0; row_idx < A->n_rows; ++row_idx) {
        int row_start = A->row_ptr[row_idx];
        int row_end = A->row_ptr[row_idx + 1] -
                      1; // Index of the last element in the current row
        int diag_j = -1; // Initialize diag_j to -1 (indicating diagonal not
                         // found yet)
        // Find the diagonal element in this row (since rows in CRS need not
        // be column-sorted)
        for (int j = row_start; j <= row_end; ++j) {
            if (A->col[j] == row_idx) {
                diag_j = j; // Store the index of the diagonal element
                D[row_idx] = A->val[j]; // Extract the diagonal value

                // Check if the diagonal value is very close to zero
                if (std::abs(D[row_idx]) < 1e-16) {
                    SanityChecker::zero_diag(
                        row_idx); // Call sanity checker for zero diagonal
                }

                if (D_inv)
                    D_inv[row_idx] = 1.0 / D[row_idx];
            }
        }

        // If no diagonal element was found for this row
        if (diag_j < 0) {
            SanityChecker::no_diag(
                row_idx); // Call sanity checker for missing diagonal
        }

        // If a diagonal element was found AND it's not already at the end
        // of the row's non-zeros, swap it into the last slot of the current
        // row's non-zero entries.
        if (diag_j >= 0 &&
            diag_j != row_end) { // Ensure diag_j is valid before swapping
            std::swap(A->col[diag_j], A->col[row_end]);
            std::swap(A->val[diag_j], A->val[row_end]);
        }
    }
}

inline void peel_diag_crs(MatrixCRS *A, double *D, double *D_inv = nullptr) {

#if 0
    peel_diag_crs_old(A, D, D_inv);
#elif 1
    peel_diag_crs_new(A, D, D_inv);
#endif
}

inline void extract_scale(MatrixCRS *A, double *D_scale = nullptr) {
#pragma omp parallel for schedule(static)
    for (int row_idx = 0; row_idx < A->n_rows; ++row_idx) {
        int row_start = A->row_ptr[row_idx];
        int row_end = A->row_ptr[row_idx + 1] - 1;
        for (int j = row_start; j <= row_end; ++j) {
            if (A->col[j] == row_idx) {

                // Check if the diagonal value is very close to zero
                if (std::abs(A->val[j]) < 1e-16) {
                    SanityChecker::zero_diag(
                        row_idx); // Call sanity checker for zero diagonal
                }

                    D_scale[row_idx] = 1.0 / std::sqrt(std::abs(A->val[j]));
            }
        }
    }
}

inline void factor_LU(Timers *timers, MatrixCRS *A, double *A_D,
                      double *A_D_inv, MatrixCRS *L, MatrixCRS *L_strict,
                      double *L_D, MatrixCRS *U, MatrixCRS *U_strict,
                      double *U_D, PrecondType preconditioner,
                      Interface *smax = nullptr) {

    split_LU(A, L, L_strict, U, U_strict);

    // NOTE: The triangular matrix we use to peel A_D must be sorted in each row
    // It's easier just to sort both L and U now, eventhough we only need A_D
    // once
    peel_diag_crs(L, A_D, A_D_inv);
    peel_diag_crs(U, A_D, A_D_inv);

    // In the case of ILU preconditioning, overwrite these with LU factors
    if (preconditioner == PrecondType::ILU0) {
        factor_ILU0(timers, A, L, L_strict, L_D, U, U_strict,
                    U_D SMAX_ARGS(smax));
        peel_diag_crs(U, U_D);

#ifdef USE_SMAX
        // NOTE: Since the SMAX library peels and stores the diagonal internally
        // We want that it peels L_D := ones(N)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < L->n_rows; ++i) {
            for (int idx = L->row_ptr[i]; idx < L->row_ptr[i + 1]; ++idx) {
                if (L->col[idx] == i) {
                    L->val[idx] = 1.0;
                    break; // done with this row
                }
            }
        }
#endif
    }
}
