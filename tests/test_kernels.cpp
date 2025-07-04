// tests/test_kernels.cpp

#include "../kernels.hpp"
#include "../sparse_matrix.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
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

// Test function for native_spmv
void test_native_spmv() {
    std::cout << "    Running native_spmv tests..." << std::endl;

    // Test Case 1: Simple 3x3 diagonal matrix
    MatrixCRS crs_mat_1(3, 3, 3);

    int row_ptr1[] = {0, 1, 2, 3};
    int col1[] = {0, 1, 2};
    double val1[] = {1.0, 2.0, 3.0};
    std::copy(row_ptr1, row_ptr1 + 4, crs_mat_1.row_ptr);
    std::copy(col1, col1 + 3, crs_mat_1.col);
    std::copy(val1, val1 + 3, crs_mat_1.val);

    std::vector<double> x1 = {1.0, 1.0, 1.0};
    std::vector<double> y1(3);
    std::vector<double> expected_y1 = {1.0, 2.0, 3.0};
    native_spmv(&crs_mat_1, x1.data(), y1.data());
    for (size_t i = 0; i < y1.size(); ++i) {
        ASSERT_NEAR(y1[i], expected_y1[i], 1e-9);
    }
    std::cout << "        native_spmv: 3x3 diagonal test passed." << std::endl;

    // Test Case 2: 3x3 dense matrix
    MatrixCRS crs_mat_2(3, 3, 9);
    int row_ptr2[] = {0, 3, 6, 9};
    int col2[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    double val2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::copy(row_ptr2, row_ptr2 + 4, crs_mat_2.row_ptr);
    std::copy(col2, col2 + 9, crs_mat_2.col);
    std::copy(val2, val2 + 9, crs_mat_2.val);

    std::vector<double> x2 = {1.0, 2.0, 3.0};
    std::vector<double> y2(3);
    std::vector<double> expected_y2 = {14.0, 32.0, 50.0};
    native_spmv(&crs_mat_2, x2.data(), y2.data());
    for (size_t i = 0; i < y2.size(); ++i) {
        ASSERT_NEAR(y2[i], expected_y2[i], 1e-9);
    }
    std::cout << "        native_spmv: 3x3 dense test passed." << std::endl;
    std::cout << "    All native_spmv tests passed." << std::endl;
}

// Test function for native_sptrsv (Forward Substitution)
void test_native_sptrsv() {
    std::cout << "    Running native_sptrsv tests..." << std::endl;

    MatrixCRS L_strict(3, 3, 3);
    int row_ptr[] = {0, 0, 1, 3};
    int col[] = {0, 0, 1};
    double val[] = {1.0, -2.0, 1.0};
    std::copy(row_ptr, row_ptr + 4, L_strict.row_ptr);
    std::copy(col, col + 3, L_strict.col);
    std::copy(val, val + 3, L_strict.val);

    std::vector<double> D = {2.0, 3.0, 4.0};
    std::vector<double> b = {2.0, 7.0, 12.0};
    std::vector<double> x(3, 0.0);
    std::vector<double> expected_x = {1.0, 2.0, 3.0};

    native_sptrsv(&L_strict, x.data(), D.data(), b.data());

    for (size_t i = 0; i < x.size(); ++i) {
        ASSERT_NEAR(x[i], expected_x[i], 1e-9);
    }
    std::cout << "        native_sptrsv: 3x3 lower-tri test passed."
              << std::endl;
    std::cout << "    All native_sptrsv tests passed." << std::endl;
}

// Test function for native_bsptrsv (Backward Substitution)
void test_native_bsptrsv() {
    std::cout << "    Running native_bsptrsv tests..." << std::endl;

    MatrixCRS U_strict(3, 3, 3);
    int row_ptr[] = {0, 2, 3, 3};
    int col[] = {1, 2, 2};
    double val[] = {1.0, -2.0, 1.0};
    std::copy(row_ptr, row_ptr + 4, U_strict.row_ptr);
    std::copy(col, col + 3, U_strict.col);
    std::copy(val, val + 3, U_strict.val);

    std::vector<double> D = {2.0, 3.0, 4.0};
    std::vector<double> b = {-2.0, 9.0, 12.0};
    std::vector<double> x(3, 0.0);
    std::vector<double> expected_x = {1.0, 2.0, 3.0};

    native_bsptrsv(&U_strict, x.data(), D.data(), b.data());

    for (size_t i = 0; i < x.size(); ++i) {
        ASSERT_NEAR(x[i], expected_x[i], 1e-9);
    }
    std::cout << "        native_bsptrsv: 3x3 upper-tri test passed."
              << std::endl;
    std::cout << "    All native_bsptrsv tests passed." << std::endl;
} // Destructor for U_strict is called here.

void test_vector_operations() {
    std::cout << "    Running vector_operations tests..." << std::endl;
    const int N = 4;
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vec2 = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> result(N);
    subtract_vectors(result.data(), vec1.data(), vec2.data(), N);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(result[i], vec1[i] - vec2[i], 1e-9);
    subtract_vectors(result.data(), vec1.data(), vec2.data(), N, 2.0);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(result[i], vec1[i] - 2.0 * vec2[i], 1e-9);
    std::cout << "        subtract_vectors tests passed." << std::endl;
    sum_vectors(result.data(), vec1.data(), vec2.data(), N);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(result[i], vec1[i] + vec2[i], 1e-9);
    sum_vectors(result.data(), vec1.data(), vec2.data(), N, 3.0);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(result[i], vec1[i] + 3.0 * vec2[i], 1e-9);
    std::cout << "        sum_vectors tests passed." << std::endl;
    double dot_res = dot(vec1.data(), vec2.data(), N);
    ASSERT_NEAR(dot_res, 25.0, 1e-9);
    std::cout << "        dot test passed." << std::endl;
    scale(result.data(), vec1.data(), 5.0, N);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(result[i], vec1[i] * 5.0, 1e-9);
    std::cout << "        scale test passed." << std::endl;
    copy_vector(result.data(), vec1.data(), N);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(result[i], vec1[i], 1e-9);
    std::cout << "        copy_vector test passed." << std::endl;
    std::cout << "    All vector_operations tests passed." << std::endl;
}

void test_apply_preconditioner() {
    std::cout << "    Running apply_preconditioner tests..." << std::endl;
    const int N = 3;
    std::vector<double> rhs = {6.0, 12.0, 20.0};
    std::vector<double> D = {2.0, 3.0, 4.0};
    std::vector<double> vec(N);
    std::vector<double> tmp(N);

    // Test None
    apply_preconditioner(PrecondType::None, N, nullptr, nullptr, nullptr,
                         vec.data(), rhs.data(), nullptr);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(vec[i], rhs[i], 1e-9);
    std::cout << "        PrecondType::None test passed." << std::endl;

    // Test Jacobi
    apply_preconditioner(PrecondType::Jacobi, N, nullptr, nullptr, D.data(),
                         vec.data(), rhs.data(), nullptr);
    for (int i = 0; i < N; ++i)
        ASSERT_NEAR(vec[i], rhs[i] / D[i], 1e-9);
    std::cout << "        PrecondType::Jacobi test passed." << std::endl;

    // Test GaussSeidel
    {
        MatrixCRS L_strict(N, N, 3);
        // ... (populate L_strict) ...
        int row_ptr[] = {0, 0, 1, 3}, col[] = {0, 0, 1};
        double val[] = {1.0, -2.0, 1.0};
        std::copy(row_ptr, row_ptr + 4, L_strict.row_ptr);
        std::copy(col, col + 3, L_strict.col);
        std::copy(val, val + 3, L_strict.val);

        std::vector<double> b_gs = {2.0, 7.0, 12.0};
        std::vector<double> expected_x_gs = {1.0, 2.0, 3.0};
        apply_preconditioner(PrecondType::GaussSeidel, N, &L_strict, nullptr,
                             D.data(), vec.data(), b_gs.data(), nullptr);
        for (int i = 0; i < N; ++i)
            ASSERT_NEAR(vec[i], expected_x_gs[i], 1e-9);
        std::cout << "        PrecondType::GaussSeidel test passed."
                  << std::endl;
    }

    // Test BackwardsGaussSeidel
    {
        MatrixCRS U_strict(N, N, 3);
        // ... (populate U_strict) ...
        int row_ptr[] = {0, 2, 3, 3}, col[] = {1, 2, 2};
        double val[] = {1.0, -2.0, 1.0};
        std::copy(row_ptr, row_ptr + 4, U_strict.row_ptr);
        std::copy(col, col + 3, U_strict.col);
        std::copy(val, val + 3, U_strict.val);

        std::vector<double> b_bgs = {-2.0, 9.0, 12.0};
        std::vector<double> expected_x_bgs = {1.0, 2.0, 3.0};
        apply_preconditioner(PrecondType::BackwardsGaussSeidel, N, nullptr,
                             &U_strict, D.data(), vec.data(), b_bgs.data(),
                             nullptr);
        for (int i = 0; i < N; ++i)
            ASSERT_NEAR(vec[i], expected_x_bgs[i], 1e-9);
        std::cout << "        PrecondType::BackwardsGaussSeidel test passed."
                  << std::endl;
    }

    std::cout << "    All apply_preconditioner tests passed." << std::endl;
}

// Register the test cases for kernels
namespace {
struct RegisterKernelsTests {
    RegisterKernelsTests() {
        register_test("Kernels::NativeSPMV", test_native_spmv);
        register_test("Kernels::NativeSPTRSV", test_native_sptrsv);
        register_test("Kernels::NativeBSPTRSV", test_native_bsptrsv);
        register_test("Kernels::VectorOperations", test_vector_operations);
        register_test("Kernels::ApplyPreconditioner",
                      test_apply_preconditioner);
    }
};
static RegisterKernelsTests reg_kernels_tests;
} // namespace
