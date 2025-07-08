#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include "../common.hpp"
#include "../sparse_matrix.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

inline void parse_cli(Args *cli_args, int argc, char *argv[],
                      bool bench_mode = false) {
    if ((argc < 2 && bench_mode) || (argc < 3 && !bench_mode)) {
        printf("ERROR: parse_cli: Not enough arguments given. A call should "
               "contain:"
               "\n%s <matrix> <method> [extra_features]\n",
               argv[0]);
        exit(EXIT_FAILURE);
    }
    cli_args->matrix_file_name = argv[1];
    int args_start_index = 2;

    if (!bench_mode) {
        // Account for mandatory solver type
        ++args_start_index;

        std::string st = argv[2];

        if (st == "-j") {
            cli_args->method = SolverType::Jacobi;
        } else if (st == "-gs") {
            cli_args->method = SolverType::GaussSeidel;
        } else if (st == "-sgs") {
            cli_args->method = SolverType::SymmetricGaussSeidel;
        } else if (st == "-cg") {
            cli_args->method = SolverType::ConjugateGradient;
        } else if (st == "-gm") {
            cli_args->method = SolverType::GMRES;
        } else if (st == "-bi") {
            cli_args->method = SolverType::BiCGSTAB;
        } else {
            printf("ERROR: parse_cli: Please choose an available solver:"
                   "\n-j (Jacobi)"
                   "\n-gs (Gauss-Seidel)"
                   "\n-sgs (Symmetric Gauss-Seidel)"
                   "\n-gm ([Preconditioned] GMRES)"
                   "\n-cg ([Preconditioned] Conjugate Gradient)"
                   "\n-bi ([Preconditioned] BiCGSTAB)\n");
            exit(EXIT_FAILURE);
        }
    }

    // Scan remaining incoming args
    for (int i = args_start_index; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p") {
            std::string pt = argv[++i];

            if (pt == "j") {
                cli_args->preconditioner = PrecondType::Jacobi;
            } else if (pt == "gs") {
                cli_args->preconditioner = PrecondType::GaussSeidel;
            } else if (pt == "bgs") {
                cli_args->preconditioner = PrecondType::BackwardsGaussSeidel;
            } else if (pt == "sgs") {
                cli_args->preconditioner = PrecondType::SymmetricGaussSeidel;
            } else if (pt == "ilut") { 
                cli_args->preconditioner = PrecondType::ILUT;
            }
            else {
                fprintf(stderr,
                        "ERROR: assign_cli_inputs: Please choose an available "
                        "preconditioner type: "
                        "\n-p j (Jacobi)"
                        "\n-p gs (Gauss-Seidel)"
                        "\n-p bgs (Backwards Gauss-Seidel)"
                        "\n-p sgs (Symmetric Gauss-Seidel)"
                        "\n-p ilut (Incomplete LU with tolerance)\n" 
                        );
                exit(EXIT_FAILURE);
            }
        }	    

    /*	
    //ILU(0) case	
    // Scan remaining incoming args
    for (int i = args_start_index; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-p") {
            if (i + 1 >= argc) {
                printf("ERROR: parse_cli: Not enough arguments given. Some "
                       "extra features"
                       " need additional arguments. Example:"
                       "\n%s <matrix> <method> -p gs\n",
                       argv[0]);
                exit(EXIT_FAILURE);
            }
            std::string pt = argv[++i];

            if (pt == "j") {
                cli_args->preconditioner = PrecondType::Jacobi;
            } else if (pt == "gs") {
                cli_args->preconditioner = PrecondType::GaussSeidel;
            } else if (pt == "bgs") {
                cli_args->preconditioner = PrecondType::BackwardsGaussSeidel;
            } else if (pt == "sgs") {
                cli_args->preconditioner = PrecondType::SymmetricGaussSeidel;
            } else if (pt == "2st") {
                cli_args->preconditioner = PrecondType::TwoStageGS;
            } else if (pt == "ilu0") {
                cli_args->preconditioner = PrecondType::ILU0;
            } else {
                fprintf(stderr,
                        "ERROR: assign_cli_inputs: Please choose an available "
                        "preconditioner type: "
                        "\n-p j (Jacobi)"
                        "\n-p gs (Gauss-Seidel)"
                        "\n-p bgs (Backwards Gauss-Seidel)"
                        "\n-p sgs (Symmetric Gauss-Seidel)"
                        "\n-p 2st (2 Stage Gauss-Seidel)"
                        "\n-p ilu0 (Incomplete LU with zero fill-in)"
                        "\n");
                exit(EXIT_FAILURE);
            }
        }
	*/	
        // TODO: reintroduce matrix scaling
        // if (arg == "-scale"){
        // 	 std::string scale = argv[++i];

        // 		if (scale == "max"){
        // 				args->scale_type = "max";
        // 		}
        // 		else if (scale == "diag"){
        // 				args->scale_type = "diag";
        // 		}
        // 		else if (scale == "none"){
        // 				args->scale_type = "none";
        // 		}
        // 		else{
        // 				printf("ERROR: assign_cli_inputs: Please
        // choose an available matrix scaling type:\nmax (Max row/col
        // element)\ndiag (Diagonal)\nnone\n");
        // exit(1);
        // 		}
        // }
        else {
            std::cout << "ERROR: assign_cli_inputs: Arguement \"" << arg
                      << "\" not recongnized." << std::endl;
        }
    }
};

inline void init_timers(Timers *timers) {
    CREATE_STOPWATCH(total)
    CREATE_STOPWATCH(preprocessing)
    CREATE_STOPWATCH(solve)
    CREATE_STOPWATCH(per_iteration)
    CREATE_STOPWATCH(iterate)
    CREATE_STOPWATCH(spmv)
    CREATE_STOPWATCH(precond)
    CREATE_STOPWATCH(dot)
    CREATE_STOPWATCH(copy1)
    CREATE_STOPWATCH(copy2)
    CREATE_STOPWATCH(normalize)
    CREATE_STOPWATCH(sum)
    CREATE_STOPWATCH(norm)
    CREATE_STOPWATCH(scale)
    CREATE_STOPWATCH(sptrsv)
    CREATE_STOPWATCH(dgemm)
    CREATE_STOPWATCH(dgemv)
    CREATE_STOPWATCH(orthog)
    CREATE_STOPWATCH(least_sq)
    CREATE_STOPWATCH(update_g)
    CREATE_STOPWATCH(sample)
    CREATE_STOPWATCH(exchange)
    CREATE_STOPWATCH(restart)
    CREATE_STOPWATCH(save_x_star)
    CREATE_STOPWATCH(postprocessing)
}

inline void print_timers(Args *cli_args, Timers *timers) {
    long double total_time = timers->total_time->get_wtime();
    long double preprocessing_time = timers->preprocessing_time->get_wtime();
    long double solve_time = timers->solve_time->get_wtime();
    long double precond_time = timers->precond_time->get_wtime();
    long double iterate_time = timers->iterate_time->get_wtime();
    long double spmv_time = timers->spmv_time->get_wtime();
    long double dgemm_time = timers->dgemm_time->get_wtime();
    long double dgemv_time = timers->dgemv_time->get_wtime();
    long double normalize_time = timers->normalize_time->get_wtime();
    long double dot_time = timers->dot_time->get_wtime();
    long double copy1_time = timers->copy1_time->get_wtime();
    long double copy2_time = timers->copy2_time->get_wtime();
    long double sum_time = timers->sum_time->get_wtime();
    long double scale_time = timers->scale_time->get_wtime();
    long double norm_time = timers->norm_time->get_wtime();
    long double sptrsv_time = timers->sptrsv_time->get_wtime();
    long double orthog_time = timers->orthog_time->get_wtime();
    long double least_sq_time = timers->least_sq_time->get_wtime();
    long double update_g_time = timers->update_g_time->get_wtime();
    long double sample_time = timers->sample_time->get_wtime();
    long double exchange_time = timers->exchange_time->get_wtime();
    long double restart_time = timers->restart_time->get_wtime();
    long double save_x_star_time = timers->save_x_star_time->get_wtime();
    long double postprocessing_time = timers->postprocessing_time->get_wtime();

    int right_flush_width = 30;
    int left_flush_width = 25;

    std::cout << std::endl;
    std::cout << std::scientific;
    std::cout << std::setprecision(3);

    // clang-format off
	{
		std::cout << "+---------------------------------------------------------+" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "Total elapsed time: " << std::right << std::setw(right_flush_width);
		std::cout << total_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| Preprocessing time: " << std::right << std::setw(right_flush_width);
		std::cout << preprocessing_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| Solve time: " << std::right << std::setw(right_flush_width);
		std::cout << solve_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | Iterate time: " << std::right << std::setw(right_flush_width);
		std::cout << iterate_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | | SpMV time: " << std::right << std::setw(right_flush_width);
		std::cout << spmv_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | | Precond. time: " << std::right << std::setw(right_flush_width);
		std::cout << precond_time  << "[s]" << std::endl;
        if(cli_args->method == SolverType::Jacobi){
			std::cout << std::left << std::setw(left_flush_width) << "| | | Normalize time: " << std::right << std::setw(right_flush_width);
			std::cout << normalize_time  << "[s]" << std::endl;
		}
		else if(cli_args->method == SolverType::GaussSeidel || cli_args->method == SolverType::SymmetricGaussSeidel){
			std::cout << std::left << std::setw(left_flush_width) << "| | | Sum time: " << std::right << std::setw(right_flush_width);
			std::cout << sum_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | SpTRSV time: " << std::right << std::setw(right_flush_width);
			std::cout << sptrsv_time  << "[s]" << std::endl;
		}
		else if(cli_args->method == SolverType::ConjugateGradient){
			std::cout << std::left << std::setw(left_flush_width) << "| | | Dot time: " << std::right << std::setw(right_flush_width);
			std::cout << dot_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | Sum time: " << std::right << std::setw(right_flush_width);
			std::cout << sum_time  << "[s]" << std::endl;
		}
		else if(cli_args->method == SolverType::GMRES){
			std::cout << std::left << std::setw(left_flush_width) << "| | | Orthog. time: " << std::right << std::setw(right_flush_width);
			std::cout << orthog_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | Dot time: " << std::right << std::setw(right_flush_width);
			std::cout << dot_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | Sum time: " << std::right << std::setw(right_flush_width);
			std::cout << sum_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | Norm time: " << std::right << std::setw(right_flush_width);
			std::cout << norm_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | Scale time: " << std::right << std::setw(right_flush_width);
			std::cout << scale_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | Least Sq. time: " << std::right << std::setw(right_flush_width);
			std::cout << least_sq_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | DGEMM time: " << std::right << std::setw(right_flush_width);
			std::cout << dgemm_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | Copy time: " << std::right << std::setw(right_flush_width);
			std::cout << copy1_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | Update g time: " << std::right << std::setw(right_flush_width);
			std::cout << update_g_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | DGEMV time: " << std::right << std::setw(right_flush_width);
			std::cout << dgemv_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | | Copy time: " << std::right << std::setw(right_flush_width);
			std::cout << copy2_time  << "[s]" << std::endl;
		}
		else if(cli_args->method == SolverType::BiCGSTAB){
			std::cout << std::left << std::setw(left_flush_width) << "| | | Dot time: " << std::right << std::setw(right_flush_width);
			std::cout << dot_time  << "[s]" << std::endl;
			std::cout << std::left << std::setw(left_flush_width) << "| | | Sum time: " << std::right << std::setw(right_flush_width);
			std::cout << sum_time  << "[s]" << std::endl;
		}
		std::cout << std::left << std::setw(left_flush_width) << "| | Sample time: " << std::right << std::setw(right_flush_width);
		std::cout << sample_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | Exchange time: " << std::right << std::setw(right_flush_width);
		std::cout << exchange_time  << "[s]" << std::endl;
		if(cli_args->method == SolverType::GMRES){
			std::cout << std::left << std::setw(left_flush_width) << "| | Restart time: " << std::right << std::setw(right_flush_width);
			std::cout << restart_time << "[s]" << std::endl;
		}
		std::cout << std::left << std::setw(left_flush_width) << "| | Save x* time: " << std::right << std::setw(right_flush_width);
		std::cout << save_x_star_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| Postprocessing time: " << std::right << std::setw(right_flush_width);
		std::cout << postprocessing_time  << "[s]" << std::endl;
		std::cout << "+---------------------------------------------------------+" << std::endl;
		std::cout << std::endl;
	}
    // clang-format on
};

inline void convert_coo_to_crs(MatrixCOO *coo_mat, MatrixCRS *crs_mat) {
    crs_mat->n_rows = coo_mat->n_rows;
    crs_mat->n_cols = coo_mat->n_cols;
    crs_mat->nnz = coo_mat->nnz;

    crs_mat->row_ptr = new int[crs_mat->n_rows + 1];
    int *nnz_per_row = new int[crs_mat->n_rows];

#ifdef USE_SMAX
    crs_mat->perm = new int[crs_mat->n_rows];
    crs_mat->inv_perm = new int[crs_mat->n_rows];
#endif

    crs_mat->col = new int[crs_mat->nnz];
    crs_mat->val = new double[crs_mat->nnz];

    for (int idx = 0; idx < crs_mat->nnz; ++idx) {
        crs_mat->col[idx] = coo_mat->J[idx];
        crs_mat->val[idx] = coo_mat->values[idx];
    }

    for (int i = 0; i < crs_mat->n_rows; ++i) {
        nnz_per_row[i] = 0;
    }

    // count nnz per row
    for (int i = 0; i < crs_mat->nnz; ++i) {
        ++nnz_per_row[coo_mat->I[i]];
    }

    crs_mat->row_ptr[0] = 0;
    for (int i = 0; i < crs_mat->n_rows; ++i) {
        crs_mat->row_ptr[i + 1] = crs_mat->row_ptr[i] + nnz_per_row[i];
    }

    if (crs_mat->row_ptr[crs_mat->n_rows] != crs_mat->nnz) {
        printf("ERROR: converting to CRS.\n");
        exit(1);
    }

    delete[] nnz_per_row;
}

// NOTE: very lazy way to do this
inline void extract_L_U(MatrixCRS *crs_mat, MatrixCRS *crs_mat_L,
                        MatrixCRS *crs_mat_L_strict, MatrixCRS *crs_mat_U,
                        MatrixCRS *crs_mat_U_strict) {
    int D_nz_count = 0;

    // Force same dimensions for consistency
    crs_mat_U->n_rows = crs_mat->n_rows;
    crs_mat_U->n_cols = crs_mat->n_cols;
    crs_mat_U->nnz = 0;
    crs_mat_U_strict->n_rows = crs_mat->n_rows;
    crs_mat_U_strict->n_cols = crs_mat->n_cols;
    crs_mat_U_strict->nnz = 0;
    crs_mat_L->n_rows = crs_mat->n_rows;
    crs_mat_L->n_cols = crs_mat->n_cols;
    crs_mat_L->nnz = 0;
    crs_mat_L_strict->n_rows = crs_mat->n_rows;
    crs_mat_L_strict->n_cols = crs_mat->n_cols;
    crs_mat_L_strict->nnz = 0;

    for (int i = 0; i < crs_mat->n_rows; ++i) {
    }

    // Count nnz
    for (int i = 0; i < crs_mat->n_rows; ++i) {
        int row_start = crs_mat->row_ptr[i];
        int row_end = crs_mat->row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = crs_mat->col[idx];

            if (col <= i) {
                ++crs_mat_L->nnz;
                if (col < i) {
                    ++crs_mat_L_strict->nnz;
                }
            }
            if (col >= i) {
                ++crs_mat_U->nnz;
                if (col > i) {
                    ++crs_mat_U_strict->nnz;
                }
            }
        }
    }

    // Allocate heap space and assign known metadata
    crs_mat_L->col = new int[crs_mat_L->nnz];
    crs_mat_L->row_ptr = new int[crs_mat->n_rows + 1];
    crs_mat_L->val = new double[crs_mat_L->nnz];
    crs_mat_L->row_ptr[0] = 0;
    crs_mat_L->n_rows = crs_mat->n_rows;
    crs_mat_L->n_cols = crs_mat->n_cols;

    crs_mat_L_strict->col = new int[crs_mat_L_strict->nnz];
    crs_mat_L_strict->row_ptr = new int[crs_mat->n_rows + 1];
    crs_mat_L_strict->val = new double[crs_mat_L_strict->nnz];
    crs_mat_L_strict->row_ptr[0] = 0;
    crs_mat_L_strict->n_rows = crs_mat->n_rows;
    crs_mat_L_strict->n_cols = crs_mat->n_cols;

    crs_mat_U->col = new int[crs_mat_U->nnz];
    crs_mat_U->row_ptr = new int[crs_mat->n_rows + 1];
    crs_mat_U->val = new double[crs_mat_U->nnz];
    crs_mat_U->row_ptr[0] = 0;
    crs_mat_U->n_rows = crs_mat->n_rows;
    crs_mat_U->n_cols = crs_mat->n_cols;

    crs_mat_U_strict->col = new int[crs_mat_U_strict->nnz];
    crs_mat_U_strict->row_ptr = new int[crs_mat->n_rows + 1];
    crs_mat_U_strict->val = new double[crs_mat_U_strict->nnz];
    crs_mat_U_strict->row_ptr[0] = 0;
    crs_mat_U_strict->n_rows = crs_mat->n_rows;
    crs_mat_U_strict->n_cols = crs_mat->n_cols;

    // Assign nonzeros
    int L_count = 0;
    int L_strict_count = 0;
    int U_count = 0;
    int U_strict_count = 0;
    for (int i = 0; i < crs_mat->n_rows; ++i) {
        int row_start = crs_mat->row_ptr[i];
        int row_end = crs_mat->row_ptr[i + 1];

        // Loop over each non-zero entry in the current row
        for (int idx = row_start; idx < row_end; ++idx) {
            int col = crs_mat->col[idx];
            double val = crs_mat->val[idx];

            if (col <= i) {
                crs_mat_L->col[L_count] = col;
                crs_mat_L->val[L_count++] = val;
                if (col < i) {
                    crs_mat_L_strict->col[L_strict_count] = col;
                    crs_mat_L_strict->val[L_strict_count++] = val;
                }
            }
            if (col >= i) {
                crs_mat_U->col[U_count] = col;
                crs_mat_U->val[U_count++] = val;
                if (col > i) {
                    crs_mat_U_strict->col[U_strict_count] = col;
                    crs_mat_U_strict->val[U_strict_count++] = val;
                }
            }
        }

        // Update row pointers
        crs_mat_L->row_ptr[i + 1] = L_count;
        crs_mat_L_strict->row_ptr[i + 1] = L_strict_count;
        crs_mat_U->row_ptr[i + 1] = U_count;
        crs_mat_U_strict->row_ptr[i + 1] = U_strict_count;
    }
}

inline void extract_D(const MatrixCOO *coo_mat, double *D,
                      bool gmres_restarted = false, bool take_sqrt = false) {
#pragma omp parallel for schedule(static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx) {
        if (coo_mat->I[nz_idx] == coo_mat->J[nz_idx]) {
            if (take_sqrt) {
                D[coo_mat->I[nz_idx]] =
                    std::sqrt(std::abs(coo_mat->values[nz_idx]));
            } else {
                D[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
            }
        }
    }
}

inline void peel_diag_crs(MatrixCRS *A, double *D, double *D_inv) {

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

// NOTE: very lazy way to do this
inline void extract_L_plus_D(MatrixCOO *coo_mat, MatrixCOO *coo_mat_L_plus_D) {
    // Force same dimensions for consistency
    coo_mat_L_plus_D->n_rows = coo_mat->n_rows;
    coo_mat_L_plus_D->n_cols = coo_mat->n_cols;
    coo_mat_L_plus_D->is_sorted = coo_mat->is_sorted;
    coo_mat_L_plus_D->is_symmetric = false;
    coo_mat_L_plus_D->nnz = 0;

    int U_nz_count = 0;

    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx) {
        // If column and row less than i, this nz is in the L_plus_D matrix
        if (coo_mat->J[nz_idx] <= coo_mat->I[nz_idx]) {
            // Copy element to lower matrix
            coo_mat_L_plus_D->I.push_back(coo_mat->I[nz_idx]);
            coo_mat_L_plus_D->J.push_back(coo_mat->J[nz_idx]);
            coo_mat_L_plus_D->values.push_back(coo_mat->values[nz_idx]);
            ++coo_mat_L_plus_D->nnz;
        } else if (coo_mat->J[nz_idx] > coo_mat->I[nz_idx]) {
            ++U_nz_count;
        } else {
            SanityChecker::print_extract_L_U_error(nz_idx);
        }
    }

    // All elements from full_coo_mtx need to be accounted for
    SanityChecker::check_copied_L_plus_D_elements(
        coo_mat->nnz, coo_mat_L_plus_D->nnz, U_nz_count);
}

// ILU(0) factorization with pivot perturbation MILU(0.
inline void compute_ilu0(const MatrixCRS *A, MatrixCRS *L_strict,
                         MatrixCRS *U_with_diag) {
    int n = A->n_rows;
    const double pivot_tolerance = 1e-8;
    const double pivot_replacement = 1e-4;

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
                if (std::abs(pivot) < pivot_tolerance) {
                    if (pivot_idx != -1) { // Pivot exists but is small
                        double pivot_sign = (pivot >= 0.0) ? 1.0 : -1.0;
                        pivot = pivot_sign * pivot_replacement;
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
    extract_L_U(&A_ilu, &temp_L, L_strict, U_with_diag, &temp_U_strict);
}

#ifdef USE_LIKWID
void register_likwid_markers() {
#pragma omp parallel
    {
        LIKWID_MARKER_REGISTER("spmv");
        LIKWID_MARKER_REGISTER("sptrsv");
        LIKWID_MARKER_REGISTER("backwards-sptrsv");
    }
}
#endif

#endif
