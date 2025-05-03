#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include "common.hpp"
#include "solver.hpp"
#include "sparse_matrix.hpp"
#include "utilities.hpp"

void preprocessing(Args *cli_args, Solver *solver, Timers *timers) {
#ifdef USE_SMAX
    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    solver->smax = smax;
#endif
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

    solver->crs_mat_L = crs_mat_L;
    solver->crs_mat_U = crs_mat_U;

#ifdef USE_SMAX
    solver->register_structs();
#endif

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