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

    // It is convenient for gauss-seidel-like methods to have
    // (strict) lower and upper triangular copies. While not
    // explicitly necessary for all methods, it's just nice to have on hand.
    MatrixCOO *coo_mat_L = new MatrixCOO;
    MatrixCOO *coo_mat_U = new MatrixCOO;
    MatrixCOO *coo_mat_L_strict = new MatrixCOO;
    MatrixCOO *coo_mat_U_strict = new MatrixCOO;
    extract_L_U(coo_mat, coo_mat_L, coo_mat_L_strict, coo_mat_U,
                coo_mat_U_strict);

    MatrixCRS *crs_mat_L = new MatrixCRS;
    MatrixCRS *crs_mat_U = new MatrixCRS;
    MatrixCRS *crs_mat_L_strict = new MatrixCRS;
    MatrixCRS *crs_mat_U_strict = new MatrixCRS;
    convert_coo_to_crs(coo_mat_L, crs_mat_L);
    convert_coo_to_crs(coo_mat_U, crs_mat_U);
    convert_coo_to_crs(coo_mat_L_strict, crs_mat_L_strict);
    convert_coo_to_crs(coo_mat_U_strict, crs_mat_U_strict);

    // NOTE: The triangular matrix we use to peel D also is sorted in each row
    // So, it's easier just to sort both L and U now
    peel_diag_crs(crs_mat_L, solver->D);
    peel_diag_crs(crs_mat_U, solver->D);

    solver->crs_mat_L = crs_mat_L;
    solver->crs_mat_U = crs_mat_U;
    solver->crs_mat_L_strict = crs_mat_L_strict;
    solver->crs_mat_U_strict = crs_mat_U_strict;

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
    delete coo_mat_L_strict;
    delete coo_mat_U_strict;
};

#endif