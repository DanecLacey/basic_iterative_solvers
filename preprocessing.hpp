#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include "common.hpp"
#include "solver.hpp"
#include "sparse_matrix.hpp"
#include "utilities/utilities.hpp"
#include "methods/ilu.hpp"

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

    std::unique_ptr<MatrixCRS> crs_mat_L = std::make_unique<MatrixCRS>();
    std::unique_ptr<MatrixCRS> crs_mat_L_strict = std::make_unique<MatrixCRS>();
    std::unique_ptr<MatrixCRS> crs_mat_U = std::make_unique<MatrixCRS>();
    std::unique_ptr<MatrixCRS> crs_mat_U_strict = std::make_unique<MatrixCRS>();

    if (solver->preconditioner == PrecondType::ILU0) {
        // For ILU0, compute the factors
        solver->L_factor = std::make_unique<MatrixCRS>();
        solver->U_factor = std::make_unique<MatrixCRS>();
        solver->D_factor_vals = std::make_unique<double[]>(solver->crs_mat->n_rows);
        
        compute_ilu0(solver->crs_mat.get(), solver->L_factor.get(), solver->U_factor.get());
        
        // The U factor contains the diagonal. Peel it off.
        peel_diag_crs(solver->U_factor.get(), solver->D_factor_vals.get());
        // L factor from extract_L_U is not strictly lower, so we are done.
    } else {
        // Standard L+D+U splitting for other preconditioners
        extract_L_U(solver->crs_mat.get(), crs_mat_L.get(), crs_mat_L_strict.get(),
                    crs_mat_U.get(), crs_mat_U_strict.get());
        peel_diag_crs(crs_mat_L.get(), solver->D);
    }

    // Collect preprocessed matrices (only if not ILU0)
    if (solver->preconditioner != PrecondType::ILU0) {
        solver->crs_mat_L = std::move(crs_mat_L);
        solver->crs_mat_U = std::move(crs_mat_U);
        solver->crs_mat_L_strict = std::move(crs_mat_L_strict);
        solver->crs_mat_U_strict = std::move(crs_mat_U_strict);
    }

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