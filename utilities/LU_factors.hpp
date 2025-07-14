#pragma once

#include "../common.hpp"
#include "../sparse_matrix.hpp"

// NOTE: very lazy way to do this
inline void split_LU(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                     MatrixCRS *U, MatrixCRS *U_strict) {
    int D_nz_count = 0;

    // Force same dimensions for consistency
    U->n_rows = A->n_rows;
    U->n_cols = A->n_cols;
    U->nnz = 0;
    U_strict->n_rows = A->n_rows;
    U_strict->n_cols = A->n_cols;
    U_strict->nnz = 0;
    L->n_rows = A->n_rows;
    L->n_cols = A->n_cols;
    L->nnz = 0;
    L_strict->n_rows = A->n_rows;
    L_strict->n_cols = A->n_cols;
    L_strict->nnz = 0;

    // Count nnz
    for (int i = 0; i < A->n_rows; ++i) {
        int row_start = A->row_ptr[i];
        int row_end = A->row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = A->col[idx];

            if (col <= i) {
                ++L->nnz;
                if (col < i) {
                    ++L_strict->nnz;
                }
            }
            if (col >= i) {
                ++U->nnz;
                if (col > i) {
                    ++U_strict->nnz;
                }
            }
        }
    }

    // Allocate heap space and assign known metadata
    L->col = new int[L->nnz];
    L->row_ptr = new int[A->n_rows + 1];
    L->val = new double[L->nnz];
    L->row_ptr[0] = 0;
    L->n_rows = A->n_rows;
    L->n_cols = A->n_cols;

    L_strict->col = new int[L_strict->nnz];
    L_strict->row_ptr = new int[A->n_rows + 1];
    L_strict->val = new double[L_strict->nnz];
    L_strict->row_ptr[0] = 0;
    L_strict->n_rows = A->n_rows;
    L_strict->n_cols = A->n_cols;

    U->col = new int[U->nnz];
    U->row_ptr = new int[A->n_rows + 1];
    U->val = new double[U->nnz];
    U->row_ptr[0] = 0;
    U->n_rows = A->n_rows;
    U->n_cols = A->n_cols;

    U_strict->col = new int[U_strict->nnz];
    U_strict->row_ptr = new int[A->n_rows + 1];
    U_strict->val = new double[U_strict->nnz];
    U_strict->row_ptr[0] = 0;
    U_strict->n_rows = A->n_rows;
    U_strict->n_cols = A->n_cols;

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

// Helper for dropping rule part 2: keeps only the p-largest elements by
// magnitude. This corresponds to the second part of the rule described in
// Saad's paper, page 6.
inline void filter_by_p_and_sort(int p,
                                 std::vector<std::pair<int, double>> &row) {
    if (row.size() > p) {
        // Find the p-th largest element efficiently without a full sort.
        std::nth_element(row.begin(), row.begin() + p, row.end(),
                         [](const auto &a, const auto &b) {
                             return std::abs(a.second) > std::abs(b.second);
                         });
        // Discard elements smaller than the p-th largest.
        row.resize(p);
    }
    // Sort the final kept elements by column index for clean CRS storage.
    std::sort(row.begin(), row.end());
}

// Implements ILUT(p, tau) based on Saad's paper,
// primarily following Algorithm 3.2 on page 5.
inline void factor_ILUT(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                        double *L_D, MatrixCRS *U, MatrixCRS *U_strict,
                        double *U_D) {
    int n = A->n_rows;

    // These will store the final factors, built row-by-row.
    std::vector<std::vector<std::pair<int, double>>> L_rows(n);
    std::vector<std::vector<std::pair<int, double>>> U_rows(n);

    // This workspace corresponds to `row(1:n)` in Saad's Algorithm 3.2.
    // It acts as a sparse accumulator for the current row `i`.
    std::vector<double> w_vals(n, 0.0);
    std::vector<int> w_indices_list;
    w_indices_list.reserve(n); // Reserve space to avoid reallocations
    std::vector<bool> w_pattern(n, false); // Fast check for non-zero existence

    // Main loop for each row `i` of the factorization.
    // Corresponds to `do i=2, n` in Algorithm 3.2, Line 1.
    for (int i = 0; i < n; ++i) {

        // --- ALGORITHM 3.2, Line 2: `row(1:n) = a(i, 1:n)` (Sparse Copy) ---
        // We "scatter" the i-th row of A into our workspace `w`.
        double row_norm = 0.0;
        for (int j_pos = A->row_ptr[i]; j_pos < A->row_ptr[i + 1]; ++j_pos) {
            int j = A->col[j_pos];
            double val = A->val[j_pos];
            w_vals[j] = val;
            if (!w_pattern[j]) { // If this is a new non-zero for this row
                w_indices_list.push_back(j);
                w_pattern[j] = true;
            }
            row_norm += val * val;
        }
        row_norm = std::sqrt(row_norm);

        // Sort indices of the current row to process columns k < i in
        // increasing order. This is necessary for the elimination loop to be
        // correct.
        std::sort(w_indices_list.begin(), w_indices_list.end());

        // --- ALGORITHM 3.2, Line 3-9: Elimination Loop ---
        // `for (k=1, i-1 and where row(k) is nonzero) do`
        int current_len = w_indices_list.size();
        for (int k_idx = 0; k_idx < current_len; ++k_idx) {
            int k = w_indices_list[k_idx];
            if (k >= i)
                break; // We are now in the U-part of the row, stop elimination.

            // --- ALGORITHM 3.2, Line 4: `row(k) := row(k) / a(k,k)` ---
            // Find pivot U(k,k) from the previously computed k-th row of U.
            double pivot = 0.0;
            for (const auto &u_entry : U_rows[k]) {
                if (u_entry.first == k) {
                    pivot = u_entry.second;
                    break;
                }
            }
            // --- CRITICAL STABILITY CHECK ---
            // If the pivot from a previous row is too small, we cannot safely
            // use this row for elimination. Skipping it is much safer than
            // creating huge numbers in the factors.
            if (std::abs(pivot) < 1e-12) {
                continue; // Skip elimination with this unstable row
            }

            double factor = w_vals[k] / pivot;

            // --- ALGORITHM 3.2, Line 5: Apply a dropping rule to `row(k)` ---
            // NOTE: This is the first of two dropping stages. Here we apply
            // a tolerance drop to the L-factor before the update.
            if (std::abs(factor) < (ILUT_TAU * row_norm)) {
                // We drop this L-factor by simply not using it for the update.
                // We also conceptually set w_vals[k] to 0 for later filtering.
                w_vals[k] = 0.0;
                continue;
            }

            // Store the final L-factor value.
            w_vals[k] = factor;

            // --- ALGORITHM 3.2, Lines 6-8: `if (row(k) != 0) then ... sparse
            // update ...` --- Update row `i` using row `k` of U.
            for (const auto &u_entry : U_rows[k]) {
                int j = u_entry.first;
                if (j > k) { // For each U(k,j)
                    // If w_vals[j] is currently zero, this update creates a new
                    // fill-in.
                    if (!w_pattern[j]) {
                        w_indices_list.push_back(j);
                        w_pattern[j] = true;
                    }
                    w_vals[j] -= factor * u_entry.second;
                }
            }
        }

        // --- ALGORITHM 3.2, Line 10-12: CORRECTED Dropping Rule Logic and copy
        // to L and U --- The order of p- and tau-dropping is critical. We first
        // select the p-largest candidates to define the structure, then apply
        // the numerical tau-drop to this smaller set of candidates.

        // STEP 1: Gather all potential candidates from the workspace without
        // any dropping.
        std::vector<std::pair<int, double>> l_row_candidates, u_row_candidates;
        double u_diag = 0.0;

        for (int col_idx : w_indices_list) {
            double val = w_vals[col_idx];
            if (col_idx < i) {
                l_row_candidates.push_back({col_idx, val});
            } else if (col_idx == i) {
                u_diag = val; // Keep diagonal separate, it's not part of the
                              // p-filter
            } else {
                u_row_candidates.push_back({col_idx, val});
            }
        }

        // STEP 2: Apply the p-filter FIRST. This determines the sparsity
        // pattern by keeping the p-largest elements by magnitude from all
        // candidates.
        filter_by_p_and_sort(ILUT_P, l_row_candidates);
        filter_by_p_and_sort(ILUT_P, u_row_candidates);

        // STEP 3: Apply the tau-filter to the SURVIVORS of the p-filter.
        // This refines the values within the chosen sparsity pattern.
        std::vector<std::pair<int, double>> l_row, u_row;
        const double drop_tol = ILUT_TAU * row_norm;

        for (const auto &entry : l_row_candidates) {
            if (std::abs(entry.second) > drop_tol) {
                l_row.push_back(entry);
            }
        }
        for (const auto &entry : u_row_candidates) {
            if (std::abs(entry.second) > drop_tol) {
                u_row.push_back(entry);
            }
        }

        // STEP 4: Handle the diagonal element. It's never dropped.
        if (std::abs(u_diag) < 1e-12) {
            u_diag = 1e-4 * row_norm; // Pivot replacement
        }
        u_row.push_back({i, u_diag});
        std::sort(u_row.begin(), u_row.end()); // Keep U row sorted

        // Store the final, correctly filtered rows. Corresponds to lines 11
        // & 12.
        L_rows[i] = l_row; // l_row is already sorted by filter_by_p_and_sort
        U_rows[i] = u_row;

        // --- ALGORITHM 3.2, Line 13: `row(1:n) = 0` (Sparse set-to-zero) ---
        // We reset only the parts of the workspace we actually used.
        for (int col_idx : w_indices_list) {
            w_vals[col_idx] = 0.0;
            w_pattern[col_idx] = false;
        }
        w_indices_list.clear();
    } // --- ALGORITHM 3.2, Line 14: `enddo` ---

    // --- Finalization: Convert vector-of-vectors to CRS format ---
    int l_nnz = 0, u_nnz = 0;
    for (int i = 0; i < n; ++i) {
        l_nnz += L_rows[i].size();
        u_nnz += U_rows[i].size();
    }

    L->n_rows = n;
    L->n_cols = n;
    L->nnz = l_nnz;
    L->row_ptr = new int[n + 1];
    L->col = new int[l_nnz];
    L->val = new double[l_nnz];
    U->n_rows = n;
    U->n_cols = n;
    U->nnz = u_nnz;
    U->row_ptr = new int[n + 1];
    U->col = new int[u_nnz];
    U->val = new double[u_nnz];

    int l_pos = 0, u_pos = 0;
    for (int i = 0; i < n; ++i) {
        L->row_ptr[i] = l_pos;
        for (const auto &entry : L_rows[i]) {
            L->col[l_pos] = entry.first;
            L->val[l_pos] = entry.second;
            l_pos++;
        }
        U->row_ptr[i] = u_pos;
        for (const auto &entry : U_rows[i]) {
            U->col[u_pos] = entry.first;
            U->val[u_pos] = entry.second;
            u_pos++;
        }
    }
    L->row_ptr[n] = l_pos;
    U->row_ptr[n] = u_pos;
}

// inline void factor_ILU0(const MatrixCRS *A, MatrixCRS *L_strict,
//                         MatrixCRS *U_with_diag) {
inline void factor_ILU0(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                        double *L_D, MatrixCRS *U, MatrixCRS *U_strict,
                        double *U_D) {
    int n = A->n_rows;

    // Create a working copy of A. We will modify this in place.
    MatrixCRS A_ilu(n, n, A->nnz);
    std::copy(A->row_ptr, A->row_ptr + n + 1, A_ilu.row_ptr);
    std::copy(A->col, A->col + A->nnz, A_ilu.col);
    std::copy(A->val, A->val + A->nnz, A_ilu.val);

    // This helper array provides fast lookups *within the current row i*.
    std::vector<int> row_lookup(n, -1);

    for (int i = 0; i < n; ++i) {
        // --- Step 1: Scatter row `i` into the lookup map ---
        for (int j_pos = A_ilu.row_ptr[i]; j_pos < A_ilu.row_ptr[i + 1];
             ++j_pos) {
            row_lookup[A_ilu.col[j_pos]] = j_pos;
        }

        // --- Step 2: Elimination loop ---
        // For each non-zero A(i,k) in the strictly lower part of row i...
        for (int k_pos = A_ilu.row_ptr[i]; k_pos < A_ilu.row_ptr[i + 1];
             ++k_pos) {
            int k = A_ilu.col[k_pos];
            if (k < i) {
                // --- THIS IS THE PIVOT LOOKUP ---
                // We need to find the diagonal element A_ilu(k,k).
                // We must search for it within row k.
                double pivot = 0.0;
                int pivot_idx = -1;
                for (int p_pos = A_ilu.row_ptr[k]; p_pos < A_ilu.row_ptr[k + 1];
                     ++p_pos) {
                    if (A_ilu.col[p_pos] == k) {
                        pivot = A_ilu.val[p_pos];
                        pivot_idx = p_pos;
                        break;
                    }
                }

                // Check for small or missing pivot
                if (std::abs(pivot) < ILU0_PIVOT_TOLERANCE) {
                    if (pivot_idx != -1) { // Pivot exists but is small
                        double pivot_sign = (pivot >= 0.0) ? 1.0 : -1.0;
                        pivot = pivot_sign * ILU0_PIVOT_REPLACEMENT;
                        A_ilu.val[pivot_idx] =
                            pivot; // Update the matrix with the new pivot
                    } else {       // Diagonal entry doesn't exist (structurally
                                   // singular)
                        continue;  // Skip this update
                    }
                }

                // Now, compute the L-factor and update the rest of row i
                double factor = A_ilu.val[k_pos] / pivot;
                A_ilu.val[k_pos] = factor; // Store L(i,k)

                // Update row `i` using row `k` of U
                // We start after the diagonal of row k
                for (int j_pos = A_ilu.row_ptr[k]; j_pos < A_ilu.row_ptr[k + 1];
                     ++j_pos) {
                    int j = A_ilu.col[j_pos];
                    if (j > k) { // For each U(k,j)
                        int ij_pos =
                            row_lookup[j]; // Check if A(i,j) exists in pattern
                        if (ij_pos != -1) {
                            A_ilu.val[ij_pos] -= factor * A_ilu.val[j_pos];
                        }
                    }
                }
            }
        }

        // --- Step 3: Gather/reset the lookup map ---
        for (int j_pos = A_ilu.row_ptr[i]; j_pos < A_ilu.row_ptr[i + 1];
             ++j_pos) {
            row_lookup[A_ilu.col[j_pos]] = -1;
        }
    }

    // --- Step 4: Split the resulting A_ilu into final L and U factors ---
    MatrixCRS temp_L, temp_U_strict;
    split_LU(&A_ilu, &temp_L, L_strict, U, &temp_U_strict);
}

inline void peel_diag_crs(MatrixCRS *A, double *D, double *D_inv = nullptr) {

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

inline void factor_LU(const MatrixCRS *A, double *A_D, double *A_D_inv,
                      MatrixCRS *L, MatrixCRS *L_strict, double *L_D,
                      MatrixCRS *U, MatrixCRS *U_strict, double *U_D,
                      PrecondType preconditioner) {

    split_LU(A, L, L_strict, U, U_strict);

    // NOTE: The triangular matrix we use to peel A_D must be sorted in each row
    // It's easier just to sort both L and U now, eventhough we only need A_D
    // once
    peel_diag_crs(L, A_D, A_D_inv);
    peel_diag_crs(U, A_D, A_D_inv);

    // In the case of ILU preconditioning, overwrite these with LU factors
    if (preconditioner == PrecondType::ILU0 ||
        preconditioner == PrecondType::ILUT) {
        if (preconditioner == PrecondType::ILU0)
            factor_ILU0(A, L, L_strict, L_D, U, U_strict, U_D);
        if (preconditioner == PrecondType::ILUT)
            factor_ILU0(A, L, L_strict, L_D, U, U_strict, U_D);

        peel_diag_crs(U, U_D);
    }
}