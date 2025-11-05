#pragma once

#include "common.hpp"
#include "solver.hpp"
#include "utilities/utilities.hpp"

void solve(Args *cli_args, Solver *solver, Timers *timers) {

    IF_DEBUG_MODE(printf("Begin solver harness\n"))
    double initial_res = solver->collected_residual_norms[0];
    bool res_3 = false;
    bool res_6 = false;
    // bool res_9 = false;

    do {
        timers->per_iteration_time->start();

        // Main solver iteration
        TIME(timers->iterate, solver->iterate(timers))

        ++solver->iter_count;

        // Sample the residual every "residual_check_len" iterations
        TIME(timers->sample,
             solver->sample_residual(timers->per_iteration_time))

        if( solver->residual_norm/initial_res < 1e-3 && !res_3){
            std::cout << "res3 => iter_count: " << solver->iter_count << std::endl;
            print_timers(cli_args, timers);
            res_3 = true;
        }

        if( solver->residual_norm/initial_res < 1e-6 && !res_6){
            std::cout << "res6 => iter_count: " << solver->iter_count << std::endl;
            print_timers(cli_args, timers);
            res_6 = true;
        }

        // if( residual_norm/initial_res < 1e-9 && !res_9){
        //     std::cout << "res9 => iter_count: " << solver->iter_count << std::endl;
        //     print_timers(cli_args, timers);
        //     res_9 = true;
        // }

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
