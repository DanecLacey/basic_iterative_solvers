#pragma once

#include "../common.hpp"
#include "../sparse_matrix.hpp"
#include "LU_factors.hpp"

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
            } else if (pt == "s2st") {
                cli_args->preconditioner = PrecondType::SymmetricTwoStageGS;
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
                        "\n-p s2st (Symmetric 2 Stage Gauss-Seidel)"
                        "\n-p ilu0 (Incomplete LU with 0 fill-in)"
                        "\n");
                exit(EXIT_FAILURE);
            }
        }
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
    CREATE_STOPWATCH(preprocessing_init)
#ifdef USE_SMAX
    CREATE_STOPWATCH(preprocessing_perm)
    CREATE_STOPWATCH(preprocessing_register)
#endif
    CREATE_STOPWATCH(preprocessing_factor)
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
    long double preprocessing_init_time =
        timers->preprocessing_init_time->get_wtime();
#ifdef USE_SMAX
    long double preprocessing_perm_time =
        timers->preprocessing_perm_time->get_wtime();
    long double preprocessing_register_time =
        timers->preprocessing_register_time->get_wtime();
#endif
    long double preprocessing_factor_time =
        timers->preprocessing_factor_time->get_wtime();
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
		std::cout << std::left << std::setw(left_flush_width) << "| | Init time: " << std::right << std::setw(right_flush_width);
		std::cout << preprocessing_init_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | Factor time: " << std::right << std::setw(right_flush_width);
		std::cout << preprocessing_factor_time  << "[s]" << std::endl;
#ifdef USE_SMAX
		std::cout << std::left << std::setw(left_flush_width) << "| | Perm time: " << std::right << std::setw(right_flush_width);
		std::cout << preprocessing_perm_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | Register time: " << std::right << std::setw(right_flush_width);
		std::cout << preprocessing_register_time  << "[s]" << std::endl;
#endif
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
