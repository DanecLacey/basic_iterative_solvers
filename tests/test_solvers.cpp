// tests/test_solvers.cpp

#include "../methods/bicgstab.hpp"
#include "../methods/cg.hpp"
#include "../methods/gauss_seidel.hpp"
#include "../methods/gmres.hpp"
#include "../methods/jacobi.hpp"
#include "../preprocessing.hpp"
#include "../solver_harness.hpp"
#include "../utilities/utilities.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// Helper for assertions
#define ASSERT_NEAR(val1, val2, tol, msg)                                      \
    do {                                                                       \
        auto _a = (val1);                                                      \
        auto _b = (val2);                                                      \
        if (std::isnan(_a) || std::isnan(_b)) {                                \
            std::cerr << "Assertion failed: " << msg                           \
                      << " - NaN detected: " << #val1 << " (" << _a << "), "   \
                      << #val2 << " (" << _b << ")"                            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("Assertion failed: NaN detected");        \
        }                                                                      \
        if (std::abs(_a - _b) > (tol)) {                                       \
            std::cerr << "Assertion failed: " << msg << " - " << #val1 << " (" \
                      << _a << ") != " << #val2 << " (" << _b                  \
                      << ") within tolerance " << (tol) << " at " << __FILE__  \
                      << ":" << __LINE__ << std::endl;                         \
            throw std::runtime_error("Assertion failed: values differ");       \
        }                                                                      \
    } while (0)

#define ASSERT_TRUE(condition, message)                                        \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "Assertion failed: " << message << " at " << __FILE__ \
                      << ":" << __LINE__ << std::endl;                         \
            throw std::runtime_error("Assertion failed");                      \
        }                                                                      \
    } while (0)

void run_solver_test(Solver *solver, const std::string &test_name) {
    std::cout << "    Running " << test_name << " test..." << std::endl;

    Args cli_args;
    Timers timers;
    init_timers(&timers);

    auto coo_mat = std::make_unique<MatrixCOO>();
    std::vector<double> b_vec, x_true;
    // A = [[2, -1, 0],
    //      [-1, 2, -1],
    //      [0, -1, 2]],
    // x_true = [1, 2, 3] -> b = [0, 0, 4]
    coo_mat->n_rows = 3;
    coo_mat->n_cols = 3;
    coo_mat->nnz = 7;
    coo_mat->I = {0, 0, 1, 1, 1, 2, 2};
    coo_mat->J = {0, 1, 0, 1, 2, 1, 2};
    coo_mat->values = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
    b_vec = {0.0, 0.0, 4.0};
    x_true = {1.0, 2.0, 3.0};

    auto A = std::make_unique<MatrixCRS>();
    convert_coo_to_crs(coo_mat.get(), A.get());

    preprocessing(&cli_args, solver, &timers, A);

    copy_vector(solver->b, b_vec.data(), b_vec.size());
    init_vector(solver->x_0, 0.0, x_true.size());

    solver->init_structs(x_true.size());
    solver->init_residual();
    solver->init_stopping_criteria();

    solve(&cli_args, solver, &timers);

    ASSERT_TRUE(solver->convergence_flag, test_name + " did not converge");
    for (size_t i = 0; i < x_true.size(); ++i) {
        ASSERT_NEAR(solver->x_star[i], x_true[i], 1e-7, test_name);
    }

    std::cout << "    " << test_name << " test passed." << std::endl;
}

void test_bicgstab_with_jacobi_on_diag_dominant_matrix() {
    std::string test_name = "BiCGSTAB w/ Jacobi (Diag Dom Matrix)";
    std::cout << "    Running " << test_name << " test..." << std::endl;

    // 1. Arrange
    Args cli_args;
    cli_args.method = SolverType::BiCGSTAB;
    cli_args.preconditioner = PrecondType::Jacobi;

    BiCGSTABSolver solver(&cli_args);
    Timers timers;
    init_timers(&timers);

    // Setup the new, well-behaved problem
    auto coo_mat = std::make_unique<MatrixCOO>();
    std::vector<double> b_vec, x_true;
    // A = [[10, -1, 0], [-1, 10, -1], [0, -1, 10]], x_true = [1, 2, 3] -> b =
    // [8, 16, 28]
    coo_mat->n_rows = 3;
    coo_mat->n_cols = 3;
    coo_mat->nnz = 7;
    coo_mat->I = {0, 0, 1, 1, 1, 2, 2};
    coo_mat->J = {0, 1, 0, 1, 2, 1, 2};
    coo_mat->values = {10.0, -1.0, -1.0, 10.0, -1.0, -1.0, 10.0};
    b_vec = {8.0, 16.0, 28.0};
    x_true = {1.0, 2.0, 3.0};

    auto A = std::make_unique<MatrixCRS>();
    convert_coo_to_crs(coo_mat.get(), A.get());

    // 2. Preprocess
    preprocessing(&cli_args, &solver, &timers, A);
    copy_vector(solver.b, b_vec.data(), b_vec.size());
    init_vector(solver.x_0, 0.0, x_true.size());
    solver.init_structs(x_true.size());
    solver.init_residual();
    solver.init_stopping_criteria();

    // 3. Act
    solve(&cli_args, &solver, &timers);

    // 4. Assert
    ASSERT_TRUE(solver.convergence_flag, test_name + " did not converge");
    for (size_t i = 0; i < x_true.size(); ++i) {
        ASSERT_NEAR(solver.x_star[i], x_true[i], 1e-7, test_name);
    }

    std::cout << "    " << test_name << " test passed." << std::endl;
}

void test_gmres_with_ilut_on_good_matrix() {
    std::string test_name = "GMRES w/ ILUT (Well-Conditioned Matrix)";
    std::cout << "    Running " << test_name << " test..." << std::endl;

    // 1. Arrange
    Args cli_args;
    cli_args.method = SolverType::GMRES;
    cli_args.preconditioner = PrecondType::ILUT;

    GMRESSolver solver(&cli_args);
    Timers timers;
    init_timers(&timers);

    // Create the well-behaved, non-symmetric test matrix directly in code.
    auto coo_mat = std::make_unique<MatrixCOO>();
    coo_mat->n_rows = 3;
    coo_mat->n_cols = 3;
    coo_mat->nnz = 8;
    coo_mat->I = {0, 0, 0, 1, 1, 1, 2, 2};
    coo_mat->J = {0, 1, 2, 0, 1, 2, 1, 2};
    coo_mat->values = {10.0, -1.0, -2.0, -3.0, 10.0, -1.0, -4.0, 10.0};

    // Define a known solution and calculate the right-hand side b = A*x
    std::vector<double> x_true = {1.0, 2.0, 3.0};
    std::vector<double> b_vec = {10 * 1 - 1 * 2 - 2 * 3,  // 10 - 2 - 6 = 2
                                 -3 * 1 + 10 * 2 - 1 * 3, // -3 + 20 - 3 = 14
                                 0 * 1 - 4 * 2 + 10 * 3}; //  0 - 8 + 30 = 22

    auto A = std::make_unique<MatrixCRS>();
    convert_coo_to_crs(coo_mat.get(), A.get());

    // 2. Preprocess (This will call your compute_ilut)
    preprocessing(&cli_args, &solver, &timers, A);

    // Overwrite b and x0 with our known values
    copy_vector(solver.b, b_vec.data(), b_vec.size());
    init_vector(solver.x_0, 0.0, x_true.size());

    solver.init_structs(x_true.size());
    solver.init_residual();
    solver.init_stopping_criteria();

    // 3. Act
    solve(&cli_args, &solver, &timers);

    // 4. Assert
    ASSERT_TRUE(solver.convergence_flag, test_name + " did not converge");
    for (size_t i = 0; i < x_true.size(); ++i) {
        ASSERT_NEAR(solver.x_star[i], x_true[i], 1e-7, test_name);
    }

    std::cout << "    " << test_name << " test passed." << std::endl;
}

// === Define Test Cases for Each Solver ===

#define DEFINE_SOLVER_TEST(TestName, SolverClass, SolverEnum, PrecondEnum)     \
    void TestName() {                                                          \
        Args cli_args;                                                         \
        cli_args.method = SolverEnum;                                          \
        cli_args.preconditioner = PrecondEnum;                                 \
        SolverClass solver(&cli_args);                                         \
        std::string name = #SolverClass;                                       \
        if (PrecondEnum != PrecondType::None) {                                \
            name += " w/ " #PrecondEnum;                                       \
        }                                                                      \
        run_solver_test(&solver, name);                                        \
    }

DEFINE_SOLVER_TEST(test_cg_solver, ConjugateGradientSolver,
                   SolverType::ConjugateGradient, PrecondType::None)
DEFINE_SOLVER_TEST(test_cg_with_jacobi_precond, ConjugateGradientSolver,
                   SolverType::ConjugateGradient, PrecondType::Jacobi)
DEFINE_SOLVER_TEST(test_bicgstab_solver, BiCGSTABSolver, SolverType::BiCGSTAB,
                   PrecondType::None)
DEFINE_SOLVER_TEST(test_bicgstab_with_jacobi_precond, BiCGSTABSolver,
                   SolverType::BiCGSTAB, PrecondType::Jacobi)
DEFINE_SOLVER_TEST(test_gmres_solver, GMRESSolver, SolverType::GMRES,
                   PrecondType::None)
DEFINE_SOLVER_TEST(test_gmres_with_jacobi_precond, GMRESSolver,
                   SolverType::GMRES, PrecondType::Jacobi)
DEFINE_SOLVER_TEST(test_jacobi_solver, JacobiSolver, SolverType::Jacobi,
                   PrecondType::None)
DEFINE_SOLVER_TEST(test_gauss_seidel_solver, GaussSeidelSolver,
                   SolverType::GaussSeidel, PrecondType::None)
DEFINE_SOLVER_TEST(test_sgs_solver, SymmetricGaussSeidelSolver,
                   SolverType::SymmetricGaussSeidel, PrecondType::None)
// DEFINE_SOLVER_TEST(test_gmres_with_ilu0_precond, GMRESSolver,
// SolverType::GMRES, PrecondType::ILUT)

// Register all the test cases
namespace {
struct RegisterSolverTests {
    RegisterSolverTests() {
        register_test("Solvers::ConjugateGradient", test_cg_solver);
        register_test("Solvers::CGWithJacobiPrecond",
                      test_cg_with_jacobi_precond);
        register_test("Solvers::BiCGSTAB", test_bicgstab_solver);
        register_test("Solvers::BiCGSTABWithJacobiPrecond",
                      test_bicgstab_with_jacobi_precond);
        // register_test("Solvers::GMRES", test_gmres_solver);
        // register_test("Solvers::GMRESWithJacobiPrecond",
        //               test_gmres_with_jacobi_precond);
        register_test("Solvers::Jacobi", test_jacobi_solver);
        register_test("Solvers::GaussSeidel", test_gauss_seidel_solver);
        register_test("Solvers::SymmetricGaussSeidel", test_sgs_solver);
        register_test("Solvers::BiCGSTABWithJacobiPrecondDD",
                      test_bicgstab_with_jacobi_on_diag_dominant_matrix);
        // register_test("Solvers::GMRES_With_ILUT_GoodMatrix",
        // test_gmres_with_ilut_on_good_matrix);
    }
};
static RegisterSolverTests reg_solver_tests;
} // namespace
