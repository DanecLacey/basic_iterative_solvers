#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include "common.hpp"
#include "solver.hpp"
#include "sparse_matrix.hpp"
#include "utilities/utilities.hpp"

void preprocessing(Args *cli_args, Solver *solver, Timers *timers,
                   std::unique_ptr<MatrixCRS> &crs_mat) {

    // Numerical preprocessing of matrix
    // TODO: Scaling options

    // Initialize structs to be used in the solver
    IF_DEBUG_MODE(printf("Initializing structs\n"))
    solver->allocate_structs(crs_mat->n_cols);
    solver->init_structs(crs_mat->n_cols);

#ifdef USE_SMAX
    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    solver->smax = smax;

    // Optionally, permute matrix for parallel SpTRSV
    if (TO_STRING(PERM_MODE) != std::string("NONE"))
        permute_mat(smax, crs_mat);
#endif

    // Collect preprocessed CRS A matrix to solver object
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
    peel_diag_crs(crs_mat_L.get(), solver->D, solver->D_inv);
    peel_diag_crs(crs_mat_U.get(), solver->D, solver->D_inv);

    // Collect preprocessed CRS L and U matrices to solver object
    solver->crs_mat_L = std::move(crs_mat_L);
    solver->crs_mat_U = std::move(crs_mat_U);
    solver->crs_mat_L_strict = std::move(crs_mat_L_strict);
    solver->crs_mat_U_strict = std::move(crs_mat_U_strict);

    // Copied from Aashutosh branch //
    solver->L_factor = std::make_unique<MatrixCRS>();
    solver->U_factor = std::make_unique<MatrixCRS>();
    solver->D_factor_vals = std::make_unique<double[]>(solver->crs_mat->n_rows);

    compute_ilu0(solver->crs_mat.get(), solver->L_factor.get(),
                 solver->U_factor.get());

#ifdef USE_SMAX
    // Register kernels and data to SMAX
    solver->register_structs();
#endif

    // Compute the initial residual vector
    IF_DEBUG_MODE(printf("Initializing residual vector\n"))
    solver->init_residual();

    // Use the initial residual and the tolerance to compute
    // the stopping criteria (tolerance * ||Ax_0 - b||_2)
    IF_DEBUG_MODE(printf("Initializing stopping criteria\n"))
    solver->init_stopping_criteria();
};

#endif