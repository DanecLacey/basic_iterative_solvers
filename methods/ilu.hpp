#pragma once

#include "../common.hpp"
#include "../sparse_matrix.hpp"
#include "../utilities/utilities.hpp" // For extract_L_U

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