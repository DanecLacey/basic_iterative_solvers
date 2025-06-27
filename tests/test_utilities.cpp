// tests/test_utilities.cpp

#include "../kernels.hpp"
#include "../utilities/utilities.hpp"
#include "test_framework.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// Helper for assertions
#define ASSERT_NEAR(val1, val2, tol)                                           \
    do {                                                                       \
        if (std::abs((val1) - (val2)) > (tol)) {                               \
            std::cerr << "Assertion failed: " << #val1 << " (" << (val1)       \
                      << ") != " << #val2 << " (" << (val2)                    \
                      << ") within tolerance " << (tol) << " at " << __FILE__  \
                      << ":" << __LINE__ << std::endl;                         \
            throw std::runtime_error("Assertion failed");                      \
        }                                                                      \
    } while (0)

#define ASSERT_EQUAL(val1, val2, name)                                         \
    do {                                                                       \
        if ((val1) != (val2)) {                                                \
            std::cerr << "Assertion failed: " << (name) << " " << #val1        \
                      << " (" << (val1) << ") != " << #val2 << " (" << (val2)  \
                      << ")"                                                   \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("Assertion failed");                      \
        }                                                                      \
    } while (0)

void test_euclidean_vec_norm() {
    std::cout << "    Running euclidean_vec_norm tests..." << std::endl;

    // Test case 1: Positive values
    std::vector<double> vec1 = {3.0, 4.0};
    double expected_norm1 = 5.0;
    double actual_norm1 = euclidean_vec_norm(vec1.data(), vec1.size());
    ASSERT_NEAR(actual_norm1, expected_norm1, 1e-9);

    // Test case 2: Mixed values
    std::vector<double> vec2 = {-1.0, 2.0, -2.0};
    double expected_norm2 = 3.0; // sqrt(1+4+4)
    double actual_norm2 = euclidean_vec_norm(vec2.data(), vec2.size());
    ASSERT_NEAR(actual_norm2, expected_norm2, 1e-9);

    // Test case 3: Zero vector
    std::vector<double> vec3(3, 0.0);
    double expected_norm3 = 0.0;
    double actual_norm3 = euclidean_vec_norm(vec3.data(), vec3.size());
    ASSERT_NEAR(actual_norm3, expected_norm3, 1e-9);

    // Test case 4: Empty vector
    std::vector<double> vec4;
    double expected_norm4 = 0.0;
    double actual_norm4 = euclidean_vec_norm(vec4.data(), vec4.size());
    ASSERT_NEAR(actual_norm4, expected_norm4, 1e-9);

    std::cout << "    All euclidean_vec_norm tests passed." << std::endl;
}

void test_convert_coo_to_crs() {
    std::cout << "    Running convert_coo_to_crs tests..." << std::endl;
    // A = [[10,  0, 20],
    //      [ 0, 30,  0],
    //      [40, 50, 60]]
    MatrixCOO coo_mat(3, 3, 6);
    coo_mat.I = {0, 0, 1, 2, 2, 2};
    coo_mat.J = {0, 2, 1, 0, 1, 2};
    coo_mat.values = {10, 20, 30, 40, 50, 60};

    MatrixCRS crs_mat;
    convert_coo_to_crs(&coo_mat, &crs_mat);

    ASSERT_EQUAL(crs_mat.n_rows, 3, "n_rows");
    ASSERT_EQUAL(crs_mat.n_cols, 3, "n_cols");
    ASSERT_EQUAL(crs_mat.nnz, 6, "nnz");

    int expected_row_ptr[] = {0, 2, 3, 6};
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQUAL(crs_mat.row_ptr[i], expected_row_ptr[i], "row_ptr");
    }

    int expected_col[] = {0, 2, 1, 0, 1, 2};
    double expected_val[] = {10, 20, 30, 40, 50, 60};
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQUAL(crs_mat.col[i], expected_col[i], "col");
        ASSERT_NEAR(crs_mat.val[i], expected_val[i], 1e-9);
    }

    std::cout << "    All convert_coo_to_crs tests passed." << std::endl;
}

void test_extract_L_U() {
    std::cout << "    Running extract_L_U tests..." << std::endl;
    // A = [[10,  1,  2],
    //      [ 3, 20,  4],
    //      [ 5,  6, 30]]
    MatrixCRS crs_mat(3, 3, 9);
    crs_mat.row_ptr = new int[4]{0, 3, 6, 9};
    crs_mat.col = new int[9]{0, 1, 2, 0, 1, 2, 0, 1, 2};
    crs_mat.val = new double[9]{10, 1, 2, 3, 20, 4, 5, 6, 30};

    MatrixCRS L, L_strict, U, U_strict;
    extract_L_U(&crs_mat, &L, &L_strict, &U, &U_strict);

    // Expected L (lower + diagonal)
    ASSERT_EQUAL(L.nnz, 6, "L.nnz");
    int exp_L_row_ptr[] = {0, 1, 3, 6};
    for (int i = 0; i < 4; ++i)
        ASSERT_EQUAL(L.row_ptr[i], exp_L_row_ptr[i], "L.row_ptr");
    int exp_L_col[] = {0, 0, 1, 0, 1, 2};
    double exp_L_val[] = {10, 3, 20, 5, 6, 30};
    for (int i = 0; i < L.nnz; ++i) {
        ASSERT_EQUAL(L.col[i], exp_L_col[i], "L.col");
        ASSERT_NEAR(L.val[i], exp_L_val[i], 1e-9);
    }
    std::cout << "        extract_L_U: L matrix test passed." << std::endl;

    // Expected L_strict (strictly lower)
    ASSERT_EQUAL(L_strict.nnz, 3, "L_strict.nnz");
    int exp_Ls_row_ptr[] = {0, 0, 1, 3};
    for (int i = 0; i < 4; ++i)
        ASSERT_EQUAL(L_strict.row_ptr[i], exp_Ls_row_ptr[i],
                     "L_strict.row_ptr");
    int exp_Ls_col[] = {0, 0, 1};
    double exp_Ls_val[] = {3, 5, 6};
    for (int i = 0; i < L_strict.nnz; ++i) {
        ASSERT_EQUAL(L_strict.col[i], exp_Ls_col[i], "L_strict.col");
        ASSERT_NEAR(L_strict.val[i], exp_Ls_val[i], 1e-9);
    }
    std::cout << "        extract_L_U: L_strict matrix test passed."
              << std::endl;

    // Expected U (upper + diagonal)
    ASSERT_EQUAL(U.nnz, 6, "U.nnz");
    int exp_U_row_ptr[] = {0, 3, 5, 6};
    for (int i = 0; i < 4; ++i)
        ASSERT_EQUAL(U.row_ptr[i], exp_U_row_ptr[i], "U.row_ptr");
    int exp_U_col[] = {0, 1, 2, 1, 2, 2};
    double exp_U_val[] = {10, 1, 2, 20, 4, 30};
    for (int i = 0; i < U.nnz; ++i) {
        ASSERT_EQUAL(U.col[i], exp_U_col[i], "U.col");
        ASSERT_NEAR(U.val[i], exp_U_val[i], 1e-9);
    }
    std::cout << "        extract_L_U: U matrix test passed." << std::endl;

    // Expected U_strict (strictly upper)
    ASSERT_EQUAL(U_strict.nnz, 3, "U_strict.nnz");
    int exp_Us_row_ptr[] = {0, 2, 3, 3};
    for (int i = 0; i < 4; ++i)
        ASSERT_EQUAL(U_strict.row_ptr[i], exp_Us_row_ptr[i],
                     "U_strict.row_ptr");
    int exp_Us_col[] = {1, 2, 2};
    double exp_Us_val[] = {1, 2, 4};
    for (int i = 0; i < U_strict.nnz; ++i) {
        ASSERT_EQUAL(U_strict.col[i], exp_Us_col[i], "U_strict.col");
        ASSERT_NEAR(U_strict.val[i], exp_Us_val[i], 1e-9);
    }
    std::cout << "        extract_L_U: U_strict matrix test passed."
              << std::endl;

    std::cout << "    All extract_L_U tests passed." << std::endl;
}

void test_peel_diag_crs() {
    std::cout << "    Running peel_diag_crs tests..." << std::endl;

    // A = [[1, 10, 2],
    //      [3, 20, 4],
    //      [30, 5, 6]]
    // Diagonals are at indices 1, 4, 6
    MatrixCRS crs_mat(3, 3, 9);
    crs_mat.row_ptr = new int[4]{0, 3, 6, 9};
    crs_mat.col = new int[9]{1, 0, 2, 0, 1, 2, 2, 0, 1};
    crs_mat.val = new double[9]{1, 10, 2, 3, 20, 4, 30, 5, 6};

    std::vector<double> D(3);
    peel_diag_crs(&crs_mat, D.data());

    // Check diagonal vector
    std::vector<double> expected_D = {10, 20, 30};
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(D[i], expected_D[i], 1e-9);
    }
    std::cout << "        peel_diag_crs: Diagonal vector extraction passed."
              << std::endl;

    // Check that the matrix was correctly modified (diagonals swapped to end of
    // row data) Expected row 0: [1, 2, 10] with cols [1, 2, 0]
    ASSERT_NEAR(crs_mat.val[2], 10.0, 1e-9);
    ASSERT_EQUAL(crs_mat.col[2], 0, "crs_mat.col[2]");

    // Expected row 1: [3, 4, 20] with cols [0, 2, 1]
    ASSERT_NEAR(crs_mat.val[5], 20.0, 1e-9);
    ASSERT_EQUAL(crs_mat.col[5], 1, "crs_mat.col[5]");

    // Expected row 2: [5, 6, 30] with cols [0, 1, 2]
    ASSERT_NEAR(crs_mat.val[8], 30.0, 1e-9);
    ASSERT_EQUAL(crs_mat.col[8], 2, "crs_mat.col[8]");

    std::cout << "        peel_diag_crs: Matrix modification passed."
              << std::endl;
    std::cout << "    All peel_diag_crs tests passed." << std::endl;
}

// Register the test cases in an anonymous namespace
namespace {
struct RegisterUtilitiesTests {
    RegisterUtilitiesTests() {
        register_test("Utilities::EuclideanVecNorm", test_euclidean_vec_norm);
        register_test("Utilities::ConvertCOOToCRS", test_convert_coo_to_crs);
        register_test("Utilities::ExtractLAndU", test_extract_L_U);
        register_test("Utilities::PeelDiagCRS", test_peel_diag_crs);
    }
};
static RegisterUtilitiesTests reg_utilities_tests;
} // namespace
//         }
//     };
//     // Create a static instance of the struct to ensure tests are registered
//     static RegisterUtilitiesTests reg_utilities_tests;
// } // namespace
