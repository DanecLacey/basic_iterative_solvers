#include "test_framework.hpp" // For the testing framework (register_test, etc.)
#include "../kernels.hpp"     
#include "../sparse_matrix.hpp" 

#include <vector>
#include <cmath>
#include <iostream>
#include <numeric> // For std::iota, std::fill

// Helper for assertions 
#define ASSERT_NEAR(val1, val2, tol) \
    do { \
        if (std::abs((val1) - (val2)) > (tol)) { \
            std::cerr << "Assertion failed: " << #val1 << " (" << (val1) << ") != " \
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

// Test function for native_spmv
void test_native_spmv() {
    std::cout << "    Running native_spmv tests..." << std::endl;

    // Test Case 1: Simple 3x3 diagonal matrix
    // A = [[1, 0, 0],
    //      [0, 2, 0],
    //      [0, 0, 3]]
    // x = [1, 1, 1]
    // y = [1, 2, 3]

    MatrixCRS crs_mat_1(3, 3, 3); // 3 rows, 3 cols, 3 non-zeros
    crs_mat_1.row_ptr = new int[4]{0, 1, 2, 3};
    crs_mat_1.col = new int[3]{0, 1, 2};
    crs_mat_1.val = new double[3]{1.0, 2.0, 3.0};

    std::vector<double> x1 = {1.0, 1.0, 1.0};
    std::vector<double> y1(3);
    std::vector<double> expected_y1 = {1.0, 2.0, 3.0};

    native_spmv(&crs_mat_1, x1.data(), y1.data());

    for (size_t i = 0; i < y1.size(); ++i) {
        ASSERT_NEAR(y1[i], expected_y1[i], 1e-9);
    }
    std::cout << "        native_spmv: 3x3 diagonal test passed." << std::endl;

    // Test Case 2: 3x3 dense matrix
    // A = [[1, 2, 3],
    //      [4, 5, 6],
    //      [7, 8, 9]]
    // x = [1, 2, 3]
    // y = [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3, 7*1 + 8*2 + 9*3]
    // y = [1+4+9, 4+10+18, 7+16+27]
    // y = [14, 32, 50]

    MatrixCRS crs_mat_2(3, 3, 9);
    crs_mat_2.row_ptr = new int[4]{0, 3, 6, 9};
    crs_mat_2.col = new int[9]{0, 1, 2, 0, 1, 2, 0, 1, 2};
    crs_mat_2.val = new double[9]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    std::vector<double> x2 = {1.0, 2.0, 3.0};
    std::vector<double> y2(3);
    std::vector<double> expected_y2 = {14.0, 32.0, 50.0};

    native_spmv(&crs_mat_2, x2.data(), y2.data());

    for (size_t i = 0; i < y2.size(); ++i) {
        ASSERT_NEAR(y2[i], expected_y2[i], 1e-9);
    }
    std::cout << "        native_spmv: 3x3 dense test passed." << std::endl;

    // Clean up dynamically allocated CRS matrix members
    //delete[] crs_mat_1.row_ptr;
    //delete[] crs_mat_1.col;
    //delete[] crs_mat_1.val;

    //delete[] crs_mat_2.row_ptr;
    //delete[] crs_mat_2.col;
    //delete[] crs_mat_2.val;

    std::cout << "    All native_spmv tests passed." << std::endl;
}

// Register the test cases for kernels
namespace {
    struct RegisterKernelsTests {
        RegisterKernelsTests() {
            register_test("Kernels::NativeSPMV", test_native_spmv);
            // We can add more test cases here
            // register_test("Kernels::ElemwiseMultVectors", test_elemwise_mult_vectors);
        }
    };
    static RegisterKernelsTests reg_kernels_tests; // Static instance to ensure registration
} // namespace