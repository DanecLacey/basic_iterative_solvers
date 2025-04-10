#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include "common.hpp"
#include "sparse_matrix.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>

void parse_cli(Args *cli_args, int argc, char *argv[], bool bench_mode = false){
	cli_args->matrix_file_name = argv[1];
	int args_start_index = 2;

	if(!bench_mode){
		// Account for manditory solver type
		++args_start_index;

		std::string st = argv[2];

		if(st == "-r"){
			cli_args->solver_type = "richardson"; 
		}
		else if(st == "-j"){
				cli_args->solver_type = "jacobi"; 
		}
		else if(st == "-gs"){
			cli_args->solver_type = "gauss-seidel";
		}
		else if(st == "-sgs"){
			cli_args->solver_type = "symmetric-gauss-seidel";
		}
		else if(st == "-cg"){
			cli_args->solver_type = "conjugate-gradient";
		}
		else if(st == "-gm"){
			cli_args->solver_type = "gmres";
		}
		else if(st == "-bi"){
			cli_args->solver_type = "bicgstab";
		}
		else{
				printf("ERROR: parse_cli: Please choose an available solver type:" \
					"\n-r (Richardson)" \
					"\n-j (Jacobi)" \
					"\n-gs (Gauss-Seidel)" \
					"\n-sgs (Symmetric Gauss-Seidel)" \
					"\n-gm ([Preconditioned] GMRES)" \
					"\n-cg ([Preconditioned] Conjugate Gradient)" \
					"\n-bi ([Preconditioned] BiCGSTAB)\n");
				exit(EXIT_FAILURE);
		}
	}
	


	// Scan remaining incoming args
	for (int i = args_start_index; i < argc; ++i){
		std::string arg = argv[i];
		if (arg == "-p"){
			std::string pt = argv[++i];

			if (pt == "j"){
				cli_args->preconditioner_type = "jacobi";
			}
			else if (pt == "gs"){
				cli_args->preconditioner_type = "gauss-seidel";
			}
			else if (pt == "bgs"){
				cli_args->preconditioner_type = "backwards-gauss-seidel";
			}
			else if (pt == "sgs"){
				cli_args->preconditioner_type = "symmetric-gauss-seidel";
			}
			else if (pt == "ffbbgs"){
				cli_args->preconditioner_type = "ffbb-gauss-seidel";
			}
			else{
				fprintf(stderr,"ERROR: assign_cli_inputs: Please choose an available preconditioner type: " \
					"\n-p j (Jacobi)" \
					"\n-p gs (Gauss-Seidel)" \
					"\n-p bgs (Backwards Gauss-Seidel)" \
					"\n-p sgs (Symmetric Gauss-Seidel)" \
					"\n-p ffbbgs (Forward-Forward-Backward-Backward Gauss-Seidel)" \
				);
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
		// 				printf("ERROR: assign_cli_inputs: Please choose an available matrix scaling type:\nmax (Max row/col element)\ndiag (Diagonal)\nnone\n");
		// 				exit(1);
		// 		}
		// }
		if (arg == "-exp"){
			std::string exp = argv[++i];

			if (exp == "spltsv_lvl"){
				cli_args->exp_kernels["spltsv"] = "lvl";
			}
			else if (exp == "spltsv_2stage"){
				cli_args->exp_kernels["spltsv"] = "2stage";
			} 
			else if (exp == "spltsv_mc"){
				cli_args->exp_kernels["spltsv"] = "mc";
			} 
			else{
				fprintf(stderr,"ERROR: assign_cli_inputs: Please choose an available experimental kernel: " \
					"\n-exp spltsv_lvl (Level-Scheduled SpLTSV)" \
					"\n-exp spltsv_2stage (2-Stage SpLTSV)" \
					"\n-exp spltsv_mc (Multicolor SpLTSV)" \
				);
				exit(EXIT_FAILURE);
			}
		}
		else{
			std::cout << "ERROR: assign_cli_inputs: Arguement \"" << arg << "\" not recongnized." << std::endl;
		}
	}
};

void init_timers(Timers *timers){
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
	CREATE_STOPWATCH(spltsv)
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

void print_timers(Args *cli_args, Timers *timers){
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
	long double spltsv_time = timers->spltsv_time->get_wtime();
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
	if(cli_args->solver_type == "richardson"){
		std::cout << std::left << std::setw(left_flush_width) << "| | | Sum time: " << std::right << std::setw(right_flush_width);
		std::cout << sum_time  << "[s]" << std::endl;
	}
	else if(cli_args->solver_type == "jacobi"){
		std::cout << std::left << std::setw(left_flush_width) << "| | | Normalize time: " << std::right << std::setw(right_flush_width);
		std::cout << normalize_time  << "[s]" << std::endl;
	}
	else if(cli_args->solver_type == "gauss-seidel"){
		std::cout << std::left << std::setw(left_flush_width) << "| | | Sum time: " << std::right << std::setw(right_flush_width);
		std::cout << sum_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | | SpLTSV time: " << std::right << std::setw(right_flush_width);
		std::cout << spltsv_time  << "[s]" << std::endl;
	}
	else if(cli_args->solver_type == "conjugate-gradient"){
		std::cout << std::left << std::setw(left_flush_width) << "| | | Dot time: " << std::right << std::setw(right_flush_width);
		std::cout << dot_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | | Sum time: " << std::right << std::setw(right_flush_width);
		std::cout << sum_time  << "[s]" << std::endl;
	}
	else if(cli_args->solver_type == "gmres"){
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
	else if(cli_args->solver_type == "bicgstab"){
		std::cout << std::left << std::setw(left_flush_width) << "| | | Dot time: " << std::right << std::setw(right_flush_width);
		std::cout << dot_time  << "[s]" << std::endl;
		std::cout << std::left << std::setw(left_flush_width) << "| | | Sum time: " << std::right << std::setw(right_flush_width);
		std::cout << sum_time  << "[s]" << std::endl;
	}
	std::cout << std::left << std::setw(left_flush_width) << "| | Sample time: " << std::right << std::setw(right_flush_width);
	std::cout << sample_time  << "[s]" << std::endl;
	std::cout << std::left << std::setw(left_flush_width) << "| | Exchange time: " << std::right << std::setw(right_flush_width);
	std::cout << exchange_time  << "[s]" << std::endl;
	if(cli_args->solver_type == "gmres"){
		std::cout << std::left << std::setw(left_flush_width) << "| | Restart time: " << std::right << std::setw(right_flush_width);
		std::cout << restart_time << "[s]" << std::endl;
	}
	std::cout << std::left << std::setw(left_flush_width) << "| | Save x* time: " << std::right << std::setw(right_flush_width);
	std::cout << save_x_star_time  << "[s]" << std::endl;
	std::cout << std::left << std::setw(left_flush_width) << "| Postprocessing time: " << std::right << std::setw(right_flush_width);
	std::cout << postprocessing_time  << "[s]" << std::endl;
	std::cout << "+---------------------------------------------------------+" << std::endl;
	std::cout << std::endl;
};

void convert_coo_to_crs(
	MatrixCOO *coo_mat,
	MatrixCRS *crs_mat
){
	crs_mat->n_rows = coo_mat->n_rows;
	crs_mat->n_cols = coo_mat->n_cols;
	crs_mat->nnz = coo_mat->nnz;

	crs_mat->row_ptr = new int [crs_mat->n_rows+1];
	int *nnz_per_row = new int [crs_mat->n_rows];

	crs_mat->col = new int [crs_mat->nnz];
	crs_mat->val = new double [crs_mat->nnz];

	for(int idx = 0; idx < crs_mat->nnz; ++idx)
	{
			crs_mat->col[idx] = coo_mat->J[idx];
			crs_mat->val[idx] = coo_mat->values[idx];
	}

	for(int i = 0; i < crs_mat->n_rows; ++i)
	{ 
			nnz_per_row[i] = 0;
	}

	//count nnz per row
	for(int i=0; i < crs_mat->nnz; ++i)
	{
			++nnz_per_row[coo_mat->I[i]];
	}

	crs_mat->row_ptr[0] = 0;
	for(int i=0; i < crs_mat->n_rows; ++i)
	{
			crs_mat->row_ptr[i+1] = crs_mat->row_ptr[i]+nnz_per_row[i];
	}

	if(crs_mat->row_ptr[crs_mat->n_rows] != crs_mat->nnz)
	{
			printf("ERROR: converting to CRS.\n");
			exit(1);
	}

	delete[] nnz_per_row;
}

// NOTE: very lazy way to do this
void extract_L_U(
	MatrixCOO *coo_mat,
	MatrixCOO *coo_mat_L,
	MatrixCOO *coo_mat_U
){
	int D_nz_count = 0;

	// Force same dimensions for consistency
	coo_mat_U->n_rows = coo_mat->n_rows;
	coo_mat_U->n_cols = coo_mat->n_cols;
	coo_mat_U->nnz = 0;
	coo_mat_U->is_sorted = coo_mat->is_sorted;
	coo_mat_U->is_symmetric = false;
	coo_mat_L->n_rows = coo_mat->n_rows;
	coo_mat_L->n_cols = coo_mat->n_cols;
	coo_mat_L->is_sorted = coo_mat->is_sorted;
	coo_mat_L->is_symmetric = false;
	coo_mat_L->nnz = 0;

	for(int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
			// If column and row less than i, this nz is in the L matrix
			if(coo_mat->J[nz_idx] < coo_mat->I[nz_idx]){
					// Copy element to lower matrix
					coo_mat_L->I.push_back(coo_mat->I[nz_idx]);
					coo_mat_L->J.push_back(coo_mat->J[nz_idx]);
					coo_mat_L->values.push_back(coo_mat->values[nz_idx]);
					++coo_mat_L->nnz;
			}
			else if(coo_mat->J[nz_idx] > coo_mat->I[nz_idx]){
					// Copy element to upper matrix
					coo_mat_U->I.push_back(coo_mat->I[nz_idx]);
					coo_mat_U->J.push_back(coo_mat->J[nz_idx]);
					coo_mat_U->values.push_back(coo_mat->values[nz_idx]);
					++coo_mat_U->nnz;
			}
			else if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
			    // Copy element to vector representing diagonal matrix
			    // NOTE: We use a different routine to extract D
					++D_nz_count;
			}
			else{
				SanityChecker::print_extract_L_U_error(nz_idx);
			}
	}

	// All elements from full_coo_mtx need to be accounted for
	SanityChecker::check_copied_L_U_elements(coo_mat->nnz, coo_mat_L->nnz, coo_mat_U->nnz, D_nz_count);
}

void extract_D(
    const MatrixCOO *coo_mat,
    double *D,
		bool gmres_restarted = false,
    bool take_sqrt = false
){
	#pragma omp parallel for schedule (static)
	for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
		if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
			if(take_sqrt){
				D[coo_mat->I[nz_idx]] = std::sqrt(std::abs(coo_mat->values[nz_idx]));
			}
			else{
				D[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
			}
		}
	}
}

// NOTE: very lazy way to do this
void extract_L_plus_D(
	MatrixCOO *coo_mat,
	MatrixCOO *coo_mat_L_plus_D
){
	// Force same dimensions for consistency
	coo_mat_L_plus_D->n_rows = coo_mat->n_rows;
	coo_mat_L_plus_D->n_cols = coo_mat->n_cols;
	coo_mat_L_plus_D->is_sorted = coo_mat->is_sorted;
	coo_mat_L_plus_D->is_symmetric = false;
	coo_mat_L_plus_D->nnz = 0;

	int U_nz_count = 0;

	for(int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
			// If column and row less than i, this nz is in the L_plus_D matrix
			if(coo_mat->J[nz_idx] <= coo_mat->I[nz_idx]){
					// Copy element to lower matrix
					coo_mat_L_plus_D->I.push_back(coo_mat->I[nz_idx]);
					coo_mat_L_plus_D->J.push_back(coo_mat->J[nz_idx]);
					coo_mat_L_plus_D->values.push_back(coo_mat->values[nz_idx]);
					++coo_mat_L_plus_D->nnz;
			}
			else if(coo_mat->J[nz_idx] > coo_mat->I[nz_idx]){
				++U_nz_count;
			}
			else{
				SanityChecker::print_extract_L_U_error(nz_idx);
			}
	}

	// All elements from full_coo_mtx need to be accounted for
	SanityChecker::check_copied_L_plus_D_elements(coo_mat->nnz, coo_mat_L_plus_D->nnz, U_nz_count);
}

#ifdef USE_LIKWID
void register_likwid_markers(){
	#pragma omp parallel
	{
		LIKWID_MARKER_REGISTER("spmv");
		LIKWID_MARKER_REGISTER("spltsv");
	}
}
#endif

#endif