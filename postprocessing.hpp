#ifndef POSTPROCESSING_HPP
#define POSTPROCESSING_HPP

#include "common.hpp"
#include "solver.hpp"
#include "sparse_matrix.hpp"

// clang-format off
void print_residuals(double *collected_residual_norms,
                     double *time_per_iteration,
                     int collected_residual_norms_count, int res_check_len) {

    std::cout << std::scientific;
    std::cout << std::setprecision(16);
    int right_flush_width = 30;
    std::string formatted_spaces0(15, ' ');
    std::string formatted_spaces1(27, ' ');
    std::string formatted_spaces2(8, ' ');

    std::cout << std::endl;
    std::cout << formatted_spaces0 << "Residual Norms" << formatted_spaces1 << "Time for iteration" << std::endl;
    std::cout << "+------------------------------------------+" << formatted_spaces2 << "+-------------------------+" << std::endl;

    for (int i = 0; i < collected_residual_norms_count; ++i) {
        std::cout << "||A*x_" << i * res_check_len << " - b||_infty = " << collected_residual_norms[i];
        if (i > 0){
            std::cout << std::right << std::setw(right_flush_width) << time_per_iteration[i + 1] << "[s]";
        }
        std::cout << std::endl;
    }
}
// clang-format on

void summary_output(Args *cli_args, Solver *solver) {

    print_residuals(
        solver->collected_residual_norms, solver->time_per_iteration,
        solver->collected_residual_norms_count, solver->residual_check_len);

    if (solver->method == SolverType::GMRES)
        solver->iter_count += solver->gmres_restart_count;

    std::cout << "\nSolver: " << to_string(solver->method);
    if (solver->preconditioner != PrecondType::None) {
        std::cout << " with preconditioner: "
                  << to_string(solver->preconditioner);
    }
    if (solver->convergence_flag) {
        // x_star = A^{-1}b
        std::cout << " converged in: " << solver->iter_count << " iterations."
                  << std::endl;
    } else {
        // x_star != A^{-1}b
        std::cout << " did not converge after " << solver->iter_count
                  << " iterations." << std::endl;
    }

    std::cout << "With the stopping criteria \"tol * ||Ax_0 - b||_infty\" is: "
              << solver->stopping_criteria << std::endl;

    std::cout
        << "The residual of the final iteration is: ||A*x_star - b||_infty = "
        << std::scientific
        << solver->collected_residual_norms
               [solver->collected_residual_norms_count - 1]
        << ".\n";
}

void postprocessing(Args *cli_args, Solver *solver, Timers *timers) {

    summary_output(cli_args, solver);

#ifdef USE_SMAX
    solver->smax->utils->print_timers();
#endif
};

#endif