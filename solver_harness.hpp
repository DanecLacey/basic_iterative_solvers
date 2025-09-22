#pragma once

#include "common.hpp"
#include "solver.hpp"

void solve(Args *cli_args, Solver *solver, Timers *timers) {

    IF_DEBUG_MODE(printf("Begin solver harness\n"))

    do {
        timers->per_iteration_time->start();

        // Main solver iteration
        TIME(timers->iterate, solver->iterate(timers))

        ++solver->iter_count;

        // Sample the residual every "residual_check_len" iterations
        TIME(timers->sample,
             solver->sample_residual(timers->per_iteration_time))

        // Swap old <-> new structs
        TIME(timers->exchange, solver->exchange())

        // Restart solver if certain conditions are met
        TIME(timers->restart, solver->check_restart(timers))

    } while (!solver->check_stopping_criteria());

    // Record if the solver converged
    if (solver->residual_norm < solver->stopping_criteria)
        solver->convergence_flag = true;

    // Record the final approximate x and residual norm
    TIME(timers->save_x_star, solver->save_x_star())

    IF_DEBUG_MODE(printf("Completing solver harness\n"))
};
