#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include "common.hpp"
#include "utilities.hpp"
#include "solver.hpp"
#include "sparse_matrix.hpp"

void preprocessing(Args *cli_args, Solver *solver, Timers *timers){
	// Read or generate matrix A to solve for x in: Ax = b
	MatrixCOO *coo_mat = new MatrixCOO;
#ifdef USE_SCAMAC
	IF_DEBUG_MODE(printf("Generating SCAMAC Matrix\n"))
	coo_mat->scamac_make_mtx(cli_args->matrix_file_name);
#else
	IF_DEBUG_MODE(printf("Reading .mtx Matrix\n"))
	coo_mat->read_from_mtx(cli_args->matrix_file_name);
#endif

	// Preprocessing on COO matrix
	// TODO: Scaling options

	// Convert A, L, and U to CRS matrices and store in solver object
	IF_DEBUG_MODE(printf("Converting COO matrix to CRS\n"))
	MatrixCRS *crs_mat = new MatrixCRS;
	convert_coo_to_crs(coo_mat, crs_mat);

	// Collect CRS matricies to solver object
	solver->crs_mat = crs_mat;

	// Initialize structs to be used in the solver
	IF_DEBUG_MODE(printf("Initializing structs\n"))
	solver->allocate_structs();
	solver->init_structs();

	// It is convenient for gauss-seidel methods to have 
	// strict lower and upper triangular copies. While not 
	// explicitly necessary for all methods, it's just nice to have.
	MatrixCOO *coo_mat_L = new MatrixCOO;
	MatrixCOO *coo_mat_U = new MatrixCOO;
	extract_L_U(coo_mat, coo_mat_L, coo_mat_U);
	extract_D(coo_mat, solver->D);

	MatrixCRS *crs_mat_L = new MatrixCRS;
	MatrixCRS *crs_mat_U = new MatrixCRS;
	convert_coo_to_crs(coo_mat_L, crs_mat_L);
	convert_coo_to_crs(coo_mat_U, crs_mat_U);

    if( cli_args->solver_type == "gauss-seidel" && cli_args->exp_kernels["spltsv"] ==  "lvl" ){
        int *levels = (int *) malloc(sizeof(int) * crs_mat_L->n_rows);
        int max_level = 0;

        // Assign initial levels for matrix structure
        for(int row_idx = 0; row_idx < crs_mat_L->n_rows; row_idx++){
            levels[row_idx] = 0;
            for (int nz_idx = crs_mat_L->row_ptr[row_idx]; nz_idx < crs_mat_L->row_ptr[row_idx + 1]; ++nz_idx) {
                levels[row_idx] = std::max(levels[row_idx], levels[crs_mat_L->col[nz_idx]] + 1);
                max_level = max_level<levels[row_idx]?levels[row_idx]:max_level;
            }
        }

        solver->level[0] = new int[max_level + 2];
        solver->level[1] = new int[crs_mat_L->n_rows];
        int *crs_level = solver->level[0];
        int *row_level = solver->level[1];

        // Initially assign zero as all levels are treated as empty
        for (int i = 0; i < max_level + 1; i++){
            crs_level[i] = 0;
        }

        // Iterate over all rows
        for (int i = 0; i < crs_mat_L->n_rows; i++){
            // Shift all further entries in the array to the right s.t. the new entry fits in
            for (int j = i; j > crs_level[levels[i]]; --j){
                row_level[j] = row_level[j - 1];
            }
            for (int j = levels[i] + 1; j < max_level + 1; j++){
                crs_level[j]++;
            }
            row_level[crs_level[levels[i]]] = i;
        }
        crs_level[max_level] = crs_mat_L->n_rows;
        crs_level[max_level + 1] = -1;
       
        // TODO: permute matrix such that we have levels ordered
        
        // TODO: print files such that we can look at them
        crs_mat->write_to_mtx_file("./crs_before");

        free(levels);
    }

	solver->crs_mat_L = crs_mat_L;
	solver->crs_mat_U = crs_mat_U;

	// Compute the initial residual vector
	IF_DEBUG_MODE(printf("Initializing residual vector\n"))
	solver->init_residual();

	// Use the initial residual and the tolerance to compute 
	// the stopping criteria (tolerance * ||Ax_0 - b||_infty)
	IF_DEBUG_MODE(printf("Initializing stopping criteria\n"))
	solver->init_stopping_criteria();

	delete coo_mat;
	delete coo_mat_L;
	delete coo_mat_U;
};

#endif
