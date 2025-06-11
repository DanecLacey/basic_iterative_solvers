#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#include "common.hpp"
#include "methods/bicgstab.hpp"
#include "methods/cg.hpp"
#include "methods/gauss_seidel.hpp"
#include "methods/gmres.hpp"
#include "methods/jacobi.hpp"
#include "postprocessing.hpp"
#include "preprocessing.hpp"
#include "solver_harness.hpp"

#include "utilities/utilities.hpp"

void run(Args *cli_args, Timers *timers) {

    // Declare solver object
    Solver *solver;

    switch (cli_args->method) {
    case SolverType::Jacobi:
        solver = new JacobiSolver(cli_args);
        break;
    case SolverType::GaussSeidel:
        solver = new GaussSeidelSolver(cli_args);
        break;
    case SolverType::SymmetricGaussSeidel:
        solver = new SymmetricGaussSeidelSolver(cli_args);
        break;
    case SolverType::ConjugateGradient:
        solver = new ConjugateGradientSolver(cli_args);
        break;
    case SolverType::GMRES:
        solver = new GMRESSolver(cli_args);
        break;
    case SolverType::BiCGSTAB:
        solver = new BiCGSTABSolver(cli_args);
        break;
    default:
        std::cerr << "Error: Unknown or unsupported solver type.\n";
        exit(EXIT_FAILURE);
    }

    // Read or generate matrix A to solve for x in: Ax = b
    std::unique_ptr<MatrixCOO> coo_mat = std::make_unique<MatrixCOO>();
#ifdef USE_SCAMAC
    IF_DEBUG_MODE(printf("Generating SCAMAC Matrix\n"))
    coo_mat->scamac_make_mtx(cli_args->matrix_file_name);
#else
    IF_DEBUG_MODE(printf("Reading .mtx Matrix\n"))
    coo_mat->read_from_mtx(cli_args->matrix_file_name);
#endif

    IF_DEBUG_MODE(printf("Converting COO matrix to CRS\n"))
    std::unique_ptr<MatrixCRS> crs_mat = std::make_unique<MatrixCRS>();
    convert_coo_to_crs(coo_mat.get(), crs_mat.get());

    // Allocate and init needed structs
    TIME(timers->preprocessing,
         preprocessing(cli_args, solver, timers, crs_mat))

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