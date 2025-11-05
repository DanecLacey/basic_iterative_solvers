#pragma once

#include "common.hpp"
#include "solver.hpp"
#include "sparse_matrix.hpp"
#include "utilities/utilities.hpp"

void preprocessing(Args *cli_args, Solver *solver, Timers *timers,
                   std::unique_ptr<MatrixCRS> &A) {

    // Numerical preprocessing of matrix
    // TODO: Scaling options

    // Initialize structs to be used in the solver
    IF_DEBUG_MODE(printf("Initializing structs\n"))
    timers->preprocessing_init_time->start();
    solver->allocate_structs(A->n_cols);
    solver->init_structs(A->n_cols);
    timers->preprocessing_init_time->stop();

    // Collect preprocessed CRS A matrix to solver object
    solver->A = std::move(A);

    if(solver->num_scale) {
        IF_DEBUG_MODE(printf("Scaling numerical problem\n"))
        // Numerical preprocessing:
        // Scale A, x_0, and b by diag(|A|) symmetrically
        // NOTE: We now are solving the system A'b' = x', where
        // A' = D^{-1/2}AD^{-1/2}, b' = D^{-1/2}b, x'= D^{-1/2}x
        int N = solver->A->n_rows;
        extract_scale(solver->A.get(), solver->A_D_scale);
        scale_mat(solver->A.get(),    solver->A_D_scale);
        scale_vec(solver->x_0,  solver->A_D_scale, N);
        scale_vec(solver->b,  solver->A_D_scale, N);
    }

#ifdef USE_SMAX
    timers->preprocessing_perm_time->start();

    // Initialize interface object
    SMAX::Interface *smax = new SMAX::Interface();
    solver->smax = smax;

    // Optionally, permute matrix for parallel SpTRSV
    if (TO_STRING(PERM_MODE) != std::string("NONE"))
        permute_mat(smax, solver->A);

    timers->preprocessing_perm_time->stop();
#endif

    // It is convenient for gauss-seidel-like methods to have
    // (strict) lower and upper triangular copies. While not
    // explicitly necessary for all methods, it's just nice to have on hand.
    solver->L = std::make_unique<MatrixCRS>();
    solver->L_strict = std::make_unique<MatrixCRS>();
    solver->U = std::make_unique<MatrixCRS>();
    solver->U_strict = std::make_unique<MatrixCRS>();

    timers->preprocessing_factor_time->start();
    // Split A into L and U copies, depending on the selected preconditioner
    factor_LU(timers, solver->A.get(), solver->A_D, solver->A_D_inv,
              solver->L.get(), solver->L_strict.get(), solver->L_D,
              solver->U.get(), solver->U_strict.get(), solver->U_D,
              solver->preconditioner SMAX_ARGS(smax));
    timers->preprocessing_factor_time->stop();

#ifdef USE_SMAX
    timers->preprocessing_register_time->start();
    // Register kernels and data to SMAX
    solver->register_structs();
    timers->preprocessing_register_time->stop();
#endif

    timers->preprocessing_init_time->start();
    // Compute the initial residual vector
    IF_DEBUG_MODE(printf("Initializing residual vector\n"))
    solver->init_residual();

    // Use the initial residual and the tolerance to compute
    // the stopping criteria (tolerance * ||Ax_0 - b||_2)
    IF_DEBUG_MODE(printf("Initializing stopping criteria\n"))
    solver->init_stopping_criteria();
    timers->preprocessing_init_time->stop();
};
