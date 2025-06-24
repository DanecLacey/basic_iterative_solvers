// tests/test_solvers.cpp

#include "test_framework.hpp"
#include "../methods/jacobi.hpp"
#include "../methods/gauss_seidel.hpp"
#include "../methods/cg.hpp"
#include "../methods/gmres.hpp"
#include "../methods/bicgstab.hpp"
#include "../preprocessing.hpp"
#include "../solver_harness.hpp"
#include "../utilities/utilities.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>

// Helper for assertions
#define ASSERT_NEAR(val1, val2, tol, msg) \
    do { \
        if (std::abs((val1) - (val2)) > (tol)) { \
            std::cerr << "Assertion failed: " << msg << " - " << #val1 << " (" << (val1) << ") != " \
                      << #val2 << " (" << (val2) << ") within tolerance " << (tol) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Assertion failed"); \
        } \
    } while(0)

#define ASSERT_TRUE(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << message \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Assertion failed"); \
        } \
    } while(0)


void run_solver_test(Solver* solver, const std::string& test_name) {
    std::cout << "    Running " << test_name << " test..." << std::endl;
    
    Args cli_args; 
    Timers timers;
    init_timers(&timers);
    
    auto coo_mat = std::make_unique<MatrixCOO>();
    std::vector<double> b_vec, x_true;
    // A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]], x_true = [1, 2, 3] -> b = [0, 0, 4]
    coo_mat->n_rows = 3; coo_mat->n_cols = 3; coo_mat->nnz = 7;
    coo_mat->I      = {0,    0,    1,    1,    1,    2,    2};
    coo_mat->J      = {0,    1,    0,    1,    2,    1,    2};
    coo_mat->values = {2.0, -1.0, -1.0,  2.0, -1.0, -1.0,  2.0};
    b_vec = {0.0, 0.0, 4.0};
    x_true = {1.0, 2.0, 3.0};

    auto crs_mat = std::make_unique<MatrixCRS>();
    convert_coo_to_crs(coo_mat.get(), crs_mat.get());
    
    preprocessing(&cli_args, solver, &timers, crs_mat);
    
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

// === Define Test Cases for Each Solver ===

#define DEFINE_SOLVER_TEST(TestName, SolverClass, SolverEnum, PrecondEnum) \
    void TestName() { \
        Args cli_args; \
        cli_args.method = SolverEnum; \
        cli_args.preconditioner = PrecondEnum; \
        SolverClass solver(&cli_args); \
        std::string name = #SolverClass; \
        if (PrecondEnum != PrecondType::None) { name += " w/ " #PrecondEnum; } \
        run_solver_test(&solver, name); \
    }


DEFINE_SOLVER_TEST(test_cg_solver, ConjugateGradientSolver, SolverType::ConjugateGradient, PrecondType::None)
DEFINE_SOLVER_TEST(test_cg_with_jacobi_precond, ConjugateGradientSolver, SolverType::ConjugateGradient, PrecondType::Jacobi)
DEFINE_SOLVER_TEST(test_bicgstab_solver, BiCGSTABSolver, SolverType::BiCGSTAB, PrecondType::None)
DEFINE_SOLVER_TEST(test_bicgstab_with_jacobi_precond, BiCGSTABSolver, SolverType::BiCGSTAB, PrecondType::Jacobi)
DEFINE_SOLVER_TEST(test_gmres_solver, GMRESSolver, SolverType::GMRES, PrecondType::None)
DEFINE_SOLVER_TEST(test_gmres_with_jacobi_precond, GMRESSolver, SolverType::GMRES, PrecondType::Jacobi)
DEFINE_SOLVER_TEST(test_jacobi_solver, JacobiSolver, SolverType::Jacobi, PrecondType::None)
DEFINE_SOLVER_TEST(test_gauss_seidel_solver, GaussSeidelSolver, SolverType::GaussSeidel, PrecondType::None)
DEFINE_SOLVER_TEST(test_sgs_solver, SymmetricGaussSeidelSolver, SolverType::SymmetricGaussSeidel, PrecondType::None)

// Register all the test cases
namespace {
    struct RegisterSolverTests {
        RegisterSolverTests() {
            register_test("Solvers::ConjugateGradient", test_cg_solver);
            register_test("Solvers::CGWithJacobiPrecond", test_cg_with_jacobi_precond);
            register_test("Solvers::BiCGSTAB", test_bicgstab_solver);
            register_test("Solvers::BiCGSTABWithJacobiPrecond", test_bicgstab_with_jacobi_precond);
            register_test("Solvers::GMRES", test_gmres_solver);
            register_test("Solvers::GMRESWithJacobiPrecond", test_gmres_with_jacobi_precond);
            register_test("Solvers::Jacobi", test_jacobi_solver);
            register_test("Solvers::GaussSeidel", test_gauss_seidel_solver);
            register_test("Solvers::SymmetricGaussSeidel", test_sgs_solver);
        }
    };
    static RegisterSolverTests reg_solver_tests;
} // namespace