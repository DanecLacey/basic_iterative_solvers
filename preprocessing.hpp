#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include "common.hpp"
#include "solver.hpp"
#include "sparse_matrix.hpp"
#include "utilities/utilities.hpp"

void preprocessing(Args *cli_args, Solver *solver, Timers *timers) {
#ifdef USE_SMAX
    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    solver->smax = smax;
#endif
    // Read or generate matrix A to solve for x in: Ax = b
    // MatrixCOO *coo_mat = new MatrixCOO;
    std::unique_ptr<MatrixCOO> coo_mat = std::make_unique<MatrixCOO>();
#ifdef USE_SCAMAC
    IF_DEBUG_MODE(printf("Generating SCAMAC Matrix\n"))
    coo_mat->scamac_make_mtx(cli_args->matrix_file_name);
#else
    IF_DEBUG_MODE(printf("Reading .mtx Matrix\n"))
    coo_mat->read_from_mtx(cli_args->matrix_file_name);
#endif

    // Preprocessing on COO matrix
    // TODO: Scaling options

    // Initialize structs to be used in the solver
    IF_DEBUG_MODE(printf("Initializing structs\n"))
    solver->allocate_structs(coo_mat->n_cols);
    solver->init_structs(coo_mat->n_cols);

    // Convert A, L, and U to CRS matrices and store in solver object
    IF_DEBUG_MODE(printf("Converting COO matrix to CRS\n"))
    std::unique_ptr<MatrixCRS> crs_mat = std::make_unique<MatrixCRS>();
    // MatrixCRS *crs_mat = new MatrixCRS;
    convert_coo_to_crs(coo_mat.get(), crs_mat.get());

#ifdef USE_SMAX
    if (TO_STRING(PERM_MODE) != std::string("NONE"))
        permute_mat(smax, crs_mat);
#endif

    // Collect CRS matrix to solver object
    solver->crs_mat = std::move(crs_mat);

    // It is convenient for gauss-seidel-like methods to have
    // (strict) lower and upper triangular copies. While not
    // explicitly necessary for all methods, it's just nice to have on hand.
    std::unique_ptr<MatrixCRS> crs_mat_L = std::make_unique<MatrixCRS>();
    std::unique_ptr<MatrixCRS> crs_mat_L_strict = std::make_unique<MatrixCRS>();
    std::unique_ptr<MatrixCRS> crs_mat_U = std::make_unique<MatrixCRS>();
    std::unique_ptr<MatrixCRS> crs_mat_U_strict = std::make_unique<MatrixCRS>();
    extract_L_U(solver->crs_mat.get(), crs_mat_L.get(), crs_mat_L_strict.get(),
                crs_mat_U.get(), crs_mat_U_strict.get());

    // NOTE: The triangular matrix we use to peel D must be sorted in each row
    // It's easier just to sort both L and U now, eventhough we only need D once
    peel_diag_crs(crs_mat_L.get(), solver->D);
    peel_diag_crs(crs_mat_U.get(), solver->D);

    solver->crs_mat_L = std::move(crs_mat_L);
    solver->crs_mat_U = std::move(crs_mat_U);
    solver->crs_mat_L_strict = std::move(crs_mat_L_strict);
    solver->crs_mat_U_strict = std::move(crs_mat_U_strict);

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
};

#endif