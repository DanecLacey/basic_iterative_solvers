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
    #pragma omp parallel for reduction(+:l_nnz, l_strict_nnz, u_nnz, u_strict_nnz) schedule(static)
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

// Implements ILUT(p, tau) based on a robust interpretation of Saad's paper.
inline void factor_ILUT(const MatrixCRS *A, MatrixCRS *L, MatrixCRS *L_strict,
                        double *L_D, MatrixCRS *U, MatrixCRS *U_strict,
                        double *U_D) {
    int n = A->n_rows;

    std::vector<std::vector<std::pair<int, double>>> L_rows(n);
    std::vector<std::vector<std::pair<int, double>>> U_rows(n);

    std::vector<double> w_vals(n, 0.0);
    std::vector<int> w_indices_list;
    w_indices_list.reserve(n);
    std::vector<bool> w_pattern(n, false);

    for (int i = 0; i < n; ++i) {
        // --- Phase 1: Elimination ---
        
        // 1a. Scatter A(i,*) into the workspace `w` and get row norm.
        double row_norm_sq = 0.0;
        for (int j_pos = A->row_ptr[i]; j_pos < A->row_ptr[i + 1]; ++j_pos) {
            int j = A->col[j_pos];
            double val = A->val[j_pos];
            w_vals[j] = val;
            if (!w_pattern[j]) {
                w_indices_list.push_back(j);
                w_pattern[j] = true;
            }
            row_norm_sq += val * val;
        }
        
        // Sort indices to ensure k is processed in increasing order.
        std::sort(w_indices_list.begin(), w_indices_list.end());

        // 1b. Elimination loop using previously computed rows of U.
        for (int k_idx = 0; k_idx < w_indices_list.size(); ++k_idx) {
            int k = w_indices_list[k_idx];
            if (k >= i) break;

            // Find pivot U(k,k) from the previously computed k-th row of U.
            double pivot = 0.0;
            for (const auto &u_entry : U_rows[k]) {
                if (u_entry.first == k) {
                    pivot = u_entry.second;
                    break;
                }
            }
            
            // This check is vital. If a previous pivot was bad, we cannot use it.
            if (std::abs(pivot) < 1e-16) { continue; }
            
            double factor = w_vals[k] / pivot;
            w_vals[k] = factor; // Store the L-factor temporarily in the workspace.

            // Sparse update: w = w - factor * U(k,*)
            for (const auto &u_entry : U_rows[k]) {
                int j = u_entry.first;
                if (j > k) { 
                    if (!w_pattern[j]) { // This update creates a new fill-in.
                        w_indices_list.push_back(j);
                        w_pattern[j] = true;
                    }
                    w_vals[j] -= factor * u_entry.second;
                }
            }
        }
        // After this loop, `w_vals` contains the fully computed, non-dropped row.

        // --- Phase 2: Dropping and Storing ---
        
        std::vector<std::pair<int, double>> l_row_final, u_row_final;
        double u_diag = 0.0;
        double drop_tol = ILUT_TAU * std::sqrt(row_norm_sq);

        // 2a. Separate `w` into L and U parts, applying tau-dropping.
        for (int col_idx : w_indices_list) {
            double val = w_vals[col_idx];
            if (std::abs(val) > drop_tol) {
                if (col_idx < i) {
                    l_row_final.push_back({col_idx, val});
                } else if (col_idx == i) {
                    u_diag = val;
                } else {
                    u_row_final.push_back({col_idx, val});
                }
            }
        }

        // 2b. Apply p-dropping to the survivors of the tau-drop.
        filter_by_p_and_sort(ILUT_P, l_row_final);
        filter_by_p_and_sort(ILUT_P, u_row_final);
        
        // 2c. Robustly handle the diagonal pivot.
        if (std::abs(u_diag) < drop_tol + 1e-16) { // Check against tolerance
            u_diag = (u_diag >= 0 ? 1.0 : -1.0) * (drop_tol + 1e-8); // Safer replacement
        }
        
        // Finalize the U part for this row
        u_row_final.push_back({i, u_diag});
        std::sort(u_row_final.begin(), u_row_final.end());
        
        L_rows[i] = l_row_final;
        U_rows[i] = u_row_final;

        // --- Phase 3: Cleanup ---
        for (int col_idx : w_indices_list) {
            w_vals[col_idx] = 0.0;
            w_pattern[col_idx] = false;
        }
        w_indices_list.clear();
    }

    // --- Finalization: Convert to CRS and set up L_D, U_D ---
    // (This part of your code was good, I'm just adding the L_D/U_D setup)
    int l_nnz = 0, u_nnz = 0;
    for(int i = 0; i < n; ++i) {
        l_nnz += L_rows[i].size();
        u_nnz += U_rows[i].size();
    }

    L_strict->n_rows = L->n_rows = n;
    L_strict->n_cols = L->n_cols = n;
    L_strict->nnz = L->nnz = l_nnz;
    L->row_ptr = new int[n + 1]; L->col = new int[l_nnz]; L->val = new double[l_nnz];
    L_strict->row_ptr = new int[n + 1]; L_strict->col = new int[l_nnz]; L_strict->val = new double[l_nnz];
    
    U_strict->n_rows = U->n_rows = n;
    U_strict->n_cols = U->n_cols = n;
    U->nnz = u_nnz;
    U->row_ptr = new int[n + 1]; U->col = new int[u_nnz]; U->val = new double[u_nnz];
    
    int l_pos = 0;
    for (int i = 0; i < n; ++i) {
        L->row_ptr[i] = l_pos;
        for (const auto &entry : L_rows[i]) {
            L->col[l_pos] = entry.first;
            L->val[l_pos] = entry.second;
            l_pos++;
        }
    }
    L->row_ptr[n] = l_pos;
    
    // Create L_strict by copying L
    std::copy(L->row_ptr, L->row_ptr + n + 1, L_strict->row_ptr);
    std::copy(L->col, L->col + l_nnz, L_strict->col);
    std::copy(L->val, L->val + l_nnz, L_strict->val);

    // Set L_D to ones for the unit-diagonal L solve
    for(int i = 0; i < n; ++i) L_D[i] = 1.0;
    
    // Populate U and peel its diagonal into U_D
    int u_pos = 0;
    for (int i = 0; i < n; ++i) {
        U->row_ptr[i] = u_pos;
        for (const auto &entry : U_rows[i]) {
            U->col[u_pos] = entry.first;
            U->val[u_pos] = entry.second;
            if (entry.first == i) {
                U_D[i] = entry.second; // Peel diagonal
            }
            u_pos++;
        }
    }
    U->row_ptr[n] = u_pos;

    // Create U_strict from U (not strictly needed by the solver, but good for consistency)
    split_LU(U, new MatrixCRS(), new MatrixCRS(), new MatrixCRS(), U_strict);
}

// Implements ILU(0) factorization without creating a full matrix copy.
// It uses a row-wise workspace, similar to the ILUT implementation.
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
        // This establishes the fixed sparsity pattern for the ILU(0) calculations.
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
            if (k >= i) break;

            // Find the pivot U(k,k) from the previously computed k-th row of U.
            double pivot = 0.0;
            for (const auto &u_entry : U_rows[k]) {
                if (u_entry.first == k) {
                    pivot = u_entry.second;
                    break;
                }
            }
            
            // Check for unstable pivot. If it's bad, we can't use this row for elimination.
            if (std::abs(pivot) < 1e-16) continue;

            // Compute the L-factor L(i,k)
            double factor = w_vals[k] / pivot;
            w_vals[k] = factor; // Store it in the workspace.

            // Perform sparse update: w(j) -= L(i,k) * U(k,j)
            // We only update elements j that are already in the sparsity pattern of row i.
            for (const auto &u_entry : U_rows[k]) {
                int j = u_entry.first;
                // Check if an element A(i,j) exists. w_vals[j] != 0 is a proxy for this.
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
    // (This code is identical to the finalization in the ILUT function)
    int l_nnz = 0, u_nnz = 0;
    for (int i = 0; i < n; ++i) {
        l_nnz += L_rows[i].size();
        u_nnz += U_rows[i].size();
    }
    
    // Allocate all required matrices
    L_strict->n_rows = L->n_rows = n;
    L_strict->n_cols = L->n_cols = n;
    L_strict->nnz = L->nnz = l_nnz;
    L->row_ptr = new int[n + 1]; L->col = new int[l_nnz]; L->val = new double[l_nnz];
    L_strict->row_ptr = new int[n + 1]; L_strict->col = new int[l_nnz]; L_strict->val = new double[l_nnz];
    
    U->n_rows = n; U->n_cols = n; U->nnz = u_nnz;
    U->row_ptr = new int[n + 1]; U->col = new int[u_nnz]; U->val = new double[u_nnz];
    
    // Populate L (which is already strictly lower)
    int pos = 0;
    for (int i = 0; i < n; ++i) {
        L->row_ptr[i] = pos;
        for (const auto &entry : L_rows[i]) {
            L->col[pos] = entry.first;
            L->val[pos] = entry.second;
            pos++;
        }
    }
    L->row_ptr[n] = pos;
    
    // L_strict is the same as L for ILU(0) as constructed
    std::copy(L->row_ptr, L->row_ptr + n + 1, L_strict->row_ptr);
    std::copy(L->col, L->col + l_nnz, L_strict->col);
    std::copy(L->val, L->val + l_nnz, L_strict->val);

    // For a unit-diagonal L-solve, L_D must be all ones.
    for (int i = 0; i < n; ++i) L_D[i] = 1.0;

    // Populate U and peel its diagonal into U_D
    pos = 0;
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
    
    // This isn't strictly needed if U_strict isn't used elsewhere, but good for completeness
    split_LU(U, new MatrixCRS(), new MatrixCRS(), new MatrixCRS(), U_strict);
}

/*
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
#ifdef USE_SMAX
    split_LU(&A_ilu, L, L_strict, U, U_strict);
#else
    // We throw away L and U_strict, since they aren't needed
    MatrixCRS temp_L, temp_U_strict;
    split_LU(&A_ilu, &temp_L, L_strict, U, &temp_U_strict);
#endif
}

*/
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
            factor_ILUT(A, L, L_strict, L_D, U, U_strict, U_D);

        peel_diag_crs(U, U_D);
#ifdef USE_SMAX
        // Set all diagonal elements to 1
        // NOTE: Since the SMAX library peels and stores the diagonal internally
        // We want that it peels L_D := ones(N)
#pragma omp parallel for
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
