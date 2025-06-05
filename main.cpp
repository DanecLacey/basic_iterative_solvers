#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#include "common.hpp"
#include "postprocessing.hpp"
#include "preprocessing.hpp"
#include "solver_harness.hpp"
#include "utilities/utilities.hpp"

void run(Args *cli_args, Timers *timers) {

    // Declare solver object
    Solver *solver = new Solver(cli_args);

    // Read in matrix from .mtx file, and allocate needed structs
    TIME(timers->preprocessing, preprocessing(cli_args, solver, timers))

    // Use the selected solver until convergence
    TIME(timers->solve, solve(cli_args, solver, timers))

    // Record solution and print summary
    TIME(timers->postprocessing, postprocessing(cli_args, solver, timers))

    delete solver;
}

int main(int argc, char *argv[]) {

#ifdef USE_LIKWID
    LIKWID_MARKER_INIT;
    register_likwid_markers();
#endif

    // Declare and initialize timers
    Timers *timers = new Timers;
    init_timers(timers);

    // Collect matrix file path and options from command line
    Args *cli_args = new Args;
    parse_cli(cli_args, argc, argv);

    // Execute all phases of the solver
    TIME(timers->total, run(cli_args, timers))

    // Print timer information to stdout
    print_timers(cli_args, timers);

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

    delete timers;
    delete cli_args;
    return 0;
}