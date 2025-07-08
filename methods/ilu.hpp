// ----------- ILUT ---------------------

#pragma once

#include "../sparse_matrix.hpp"
#include "../common.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// Saad's paper used for the ILUT algorithm: 
// https://www-users.cse.umn.edu/~saad/PDF/umsi-92-38.pdf

namespace 
{ 

// Helper for dropping rule part 2: keeps only the p-largest elements by magnitude.
// This corresponds to the second part of the rule described in Saad's paper, page 6.
void filter_by_p_and_sort(int p, std::vector<std::pair<int, double>>& row) {
    if (row.size() > p) {
        // Find the p-th largest element efficiently without a full sort.
        std::nth_element(row.begin(), row.begin() + p, row.end(),
            [](const auto& a, const auto& b) {
                return std::abs(a.second) > std::abs(b.second);
            });
        // Discard elements smaller than the p-th largest.
        row.resize(p);
    }
    // Sort the final kept elements by column index for clean CRS storage.
    std::sort(row.begin(), row.end());
}

} 


// Implements ILUT(p, tau) based on Saad's paper,
// primarily following Algorithm 3.2 on page 5.
inline void compute_ilut(const MatrixCRS* A, int p, double tau,
                         MatrixCRS* L_factor, MatrixCRS* U_factor) {
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
        for (int j_pos = A->row_ptr[i]; j_pos < A->row_ptr[i+1]; ++j_pos) {
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
        
        // Sort indices of the current row to process columns k < i in increasing order.
        // This is necessary for the elimination loop to be correct.
        std::sort(w_indices_list.begin(), w_indices_list.end());

        // --- ALGORITHM 3.2, Line 3-9: Elimination Loop ---
        // `for (k=1, i-1 and where row(k) is nonzero) do`
        int current_len = w_indices_list.size();
        for (int k_idx = 0; k_idx < current_len; ++k_idx) {
            int k = w_indices_list[k_idx];
            if (k >= i) break; // We are now in the U-part of the row, stop elimination.

            // --- ALGORITHM 3.2, Line 4: `row(k) := row(k) / a(k,k)` ---
            // Find pivot U(k,k) from the previously computed k-th row of U.
            double pivot = 0.0;
            for(const auto& u_entry : U_rows[k]) {
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
            if (std::abs(factor) < (tau * row_norm)) {
                // We drop this L-factor by simply not using it for the update.
                // We also conceptually set w_vals[k] to 0 for later filtering.
                w_vals[k] = 0.0;
                continue;
            }
            
            // Store the final L-factor value.
            w_vals[k] = factor;

            // --- ALGORITHM 3.2, Lines 6-8: `if (row(k) != 0) then ... sparse update ...` ---
            // Update row `i` using row `k` of U.
            for (const auto& u_entry : U_rows[k]) {
                int j = u_entry.first;
                if (j > k) { // For each U(k,j)
                    // If w_vals[j] is currently zero, this update creates a new fill-in.
                    if (!w_pattern[j]) {
                        w_indices_list.push_back(j);
                        w_pattern[j] = true;
                    }
                    w_vals[j] -= factor * u_entry.second;
                }
            }
        }

        // --- ALGORITHM 3.2, Line 10-12: CORRECTED Dropping Rule Logic and copy to L and U ---
        // The order of p- and tau-dropping is critical. We first select the
        // p-largest candidates to define the structure, then apply the
        // numerical tau-drop to this smaller set of candidates.
        
        // STEP 1: Gather all potential candidates from the workspace without any dropping.
        std::vector<std::pair<int, double>> l_row_candidates, u_row_candidates;
        double u_diag = 0.0;

        for (int col_idx : w_indices_list) {
            double val = w_vals[col_idx];
            if (col_idx < i) {
                l_row_candidates.push_back({col_idx, val});
            } else if (col_idx == i) {
                u_diag = val; // Keep diagonal separate, it's not part of the p-filter
            } else {
                u_row_candidates.push_back({col_idx, val});
            }
        }

        // STEP 2: Apply the p-filter FIRST. This determines the sparsity pattern
        // by keeping the p-largest elements by magnitude from all candidates.
        filter_by_p_and_sort(p, l_row_candidates);
        filter_by_p_and_sort(p, u_row_candidates);

        // STEP 3: Apply the tau-filter to the SURVIVORS of the p-filter.
        // This refines the values within the chosen sparsity pattern.
        std::vector<std::pair<int, double>> l_row, u_row;
        const double drop_tol = tau * row_norm;
        
        for (const auto& entry : l_row_candidates) {
            if (std::abs(entry.second) > drop_tol) {
                l_row.push_back(entry);
            }
        }
        for (const auto& entry : u_row_candidates) {
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

        // Store the final, correctly filtered rows. Corresponds to lines 11 & 12.
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
    for(int i=0; i<n; ++i) { l_nnz += L_rows[i].size(); u_nnz += U_rows[i].size(); }

    L_factor->n_rows = n; L_factor->n_cols = n; L_factor->nnz = l_nnz;
    L_factor->row_ptr = new int[n + 1]; L_factor->col = new int[l_nnz]; L_factor->val = new double[l_nnz];
    U_factor->n_rows = n; U_factor->n_cols = n; U_factor->nnz = u_nnz;
    U_factor->row_ptr = new int[n + 1]; U_factor->col = new int[u_nnz]; U_factor->val = new double[u_nnz];

    int l_pos = 0, u_pos = 0;
    for (int i = 0; i < n; ++i) {
        L_factor->row_ptr[i] = l_pos;
        for (const auto& entry : L_rows[i]) {
            L_factor->col[l_pos] = entry.first;
            L_factor->val[l_pos] = entry.second;
            l_pos++;
        }
        U_factor->row_ptr[i] = u_pos;
        for (const auto& entry : U_rows[i]) {
            U_factor->col[u_pos] = entry.first;
            U_factor->val[u_pos] = entry.second;
            u_pos++;
        }
    }
    L_factor->row_ptr[n] = l_pos;
    U_factor->row_ptr[n] = u_pos;
}

// ------------ ILU(0) below. Need to put ILUT in another .cpp file

//#pragma once

//#include "../common.hpp"
//#include "../sparse_matrix.hpp"
//#include "../utilities/utilities.hpp" // For extract_L_U

// // ILU(0) factorization with pivot perturbation MILU(0.
// inline void compute_ilu0(const MatrixCRS *A, MatrixCRS *L_strict,
//                          MatrixCRS *U_with_diag) {
//     int n = A->n_rows;
//     const double pivot_tolerance = 1e-8;
//     const double pivot_replacement = 1e-4;

//     // Create a working copy of A. We will modify this in place.
//     MatrixCRS A_ilu(n, n, A->nnz);
//     std::copy(A->row_ptr, A->row_ptr + n + 1, A_ilu.row_ptr);
//     std::copy(A->col, A->col + A->nnz, A_ilu.col);
//     std::copy(A->val, A->val + A->nnz, A_ilu.val);

//     // This helper array provides fast lookups *within the current row i*.
//     std::vector<int> row_lookup(n, -1);

//     for (int i = 0; i < n; ++i) {
//         // --- Step 1: Scatter row `i` into the lookup map ---
//         for (int j_pos = A_ilu.row_ptr[i]; j_pos < A_ilu.row_ptr[i + 1];
//              ++j_pos) {
//             row_lookup[A_ilu.col[j_pos]] = j_pos;
//         }

//         // --- Step 2: Elimination loop ---
//         // For each non-zero A(i,k) in the strictly lower part of row i...
//         for (int k_pos = A_ilu.row_ptr[i]; k_pos < A_ilu.row_ptr[i + 1];
//              ++k_pos) {
//             int k = A_ilu.col[k_pos];
//             if (k < i) {
//                 // --- THIS IS THE PIVOT LOOKUP ---
//                 // We need to find the diagonal element A_ilu(k,k).
//                 // We must search for it within row k.
//                 double pivot = 0.0;
//                 int pivot_idx = -1;
//                 for (int p_pos = A_ilu.row_ptr[k]; p_pos < A_ilu.row_ptr[k +
//                 1];
//                      ++p_pos) {
//                     if (A_ilu.col[p_pos] == k) {
//                         pivot = A_ilu.val[p_pos];
//                         pivot_idx = p_pos;
//                         break;
//                     }
//                 }

//                 // Check for small or missing pivot
//                 if (std::abs(pivot) < pivot_tolerance) {
//                     if (pivot_idx != -1) { // Pivot exists but is small
//                         double pivot_sign = (pivot >= 0.0) ? 1.0 : -1.0;
//                         pivot = pivot_sign * pivot_replacement;
//                         A_ilu.val[pivot_idx] =
//                             pivot; // Update the matrix with the new pivot
//                     } else {       // Diagonal entry doesn't exist
//                     (structurally
//                                    // singular)
//                         continue;  // Skip this update
//                     }
//                 }

//                 // Now, compute the L-factor and update the rest of row i
//                 double factor = A_ilu.val[k_pos] / pivot;
//                 A_ilu.val[k_pos] = factor; // Store L(i,k)

//                 // Update row `i` using row `k` of U
//                 // We start after the diagonal of row k
//                 for (int j_pos = A_ilu.row_ptr[k]; j_pos < A_ilu.row_ptr[k +
//                 1];
//                      ++j_pos) {
//                     int j = A_ilu.col[j_pos];
//                     if (j > k) { // For each U(k,j)
//                         int ij_pos =
//                             row_lookup[j]; // Check if A(i,j) exists in
//                             pattern
//                         if (ij_pos != -1) {
//                             A_ilu.val[ij_pos] -= factor * A_ilu.val[j_pos];
//                         }
//                     }
//                 }
//             }
//         }

//         // --- Step 3: Gather/reset the lookup map ---
//         for (int j_pos = A_ilu.row_ptr[i]; j_pos < A_ilu.row_ptr[i + 1];
//              ++j_pos) {
//             row_lookup[A_ilu.col[j_pos]] = -1;
//         }
//     }

//     // --- Step 4: Split the resulting A_ilu into final L and U factors ---
//     MatrixCRS temp_L, temp_U_strict;
//     extract_L_U(&A_ilu, &temp_L, L_strict, U_with_diag, &temp_U_strict);
// }
