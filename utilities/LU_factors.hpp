#pragma once

#include "../common.hpp"
#include "../sparse_matrix.hpp"

// NOTE: very lazy way to do this
inline void split_LU(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
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

// Implements ILU(0) factorization without creating a full matrix copy.
inline void factor_ILU0(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                        double *L_D, MatrixCRS *U, MatrixCRS *U_strict,
                        double *U_D) {
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
        // This establishes the fixed sparsity pattern for the ILU(0)
        // calculations.
        for (int j_pos = A->row_ptr[i]; j_pos < A->row_ptr[i + 1]; ++j_pos) {
            int j = A->col[j_pos];
            w_vals[j] = A->val[j_pos];
            w_indices.push_back(j);
        }
        // Sort indices to process dependencies (k) in increasing order.
        std::sort(w_indices.begin(), w_indices.end());

        // --- Step 2: Elimination loop ---
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

        // --- Step 3: Gather the computed row into final L and U structures ---
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
            u_diag = (u_diag >= 0 ? 1.0 : -1.0) * ILU0_PIVOT_REPLACEMENT;
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

    // --- Finalization: Convert vector-of-vectors to CRS format ---
    // Count strict nnz and how many diagonals are missing
    int l_nnz_strict = 0;
    int missing_diags = 0;
    int u_nnz = 0;
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

    // This isn't strictly needed if U_strict isn't used elsewhere, but good for
    // completeness
    split_LU(U, new MatrixCRS(), new MatrixCRS(), new MatrixCRS(), U_strict);
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

inline void factor_LU(MatrixCRS *A, double *A_D, double *A_D_inv, MatrixCRS *L,
                      MatrixCRS *L_strict, double *L_D, MatrixCRS *U,
                      MatrixCRS *U_strict, double *U_D,
                      PrecondType preconditioner) {

    split_LU(A, L, L_strict, U, U_strict);

    // NOTE: The triangular matrix we use to peel A_D must be sorted in each row
    // It's easier just to sort both L and U now, eventhough we only need A_D
    // once
    peel_diag_crs(L, A_D, A_D_inv);
    peel_diag_crs(U, A_D, A_D_inv);

    // In the case of ILU preconditioning, overwrite these with LU factors
    if (preconditioner == PrecondType::ILU0) {
        if (preconditioner == PrecondType::ILU0)
            factor_ILU0(A, L, L_strict, L_D, U, U_strict, U_D);

        peel_diag_crs(U, U_D);
#ifdef USE_SMAX
        // Set all diagonal elements to 1
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
