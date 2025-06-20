// This file contains utility function tests.
// It uses the test framework defined in test_framework.hpp.
#include "test_framework.hpp" 
#include "../utilities/utilities.hpp" 
#include "kernels.hpp"
#include <vector>
#include <cmath>
#include <cassert> 
#include <iostream>
#include <stdexcept> // For std::runtime_error used by ASSERT_NEAR

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

void test_euclidean_vec_norm() {
    std::cout << "    Running euclidean_vec_norm tests..." << std::endl;

    // Test case 1: Positive values
    std::vector<double> vec1 = {3.0, 4.0};
    double expected_norm1 = 5.0;
    double actual_norm1 = euclidean_vec_norm(vec1.data(), vec1.size());
    ASSERT_NEAR(actual_norm1, expected_norm1, 1e-9);

    // Test case 2: Mixed values
    std::vector<double> vec2 = {-1.0, 2.0, -2.0};
    double expected_norm2 = std::sqrt(1.0 + 4.0 + 4.0); // sqrt(9) = 3.0
    double actual_norm2 = euclidean_vec_norm(vec2.data(), vec2.size());
    ASSERT_NEAR(actual_norm2, expected_norm2, 1e-9);

    // Test case 3: Zero vector
    std::vector<double> vec3 = {0.0, 0.0, 0.0};
    double expected_norm3 = 0.0;
    double actual_norm3 = euclidean_vec_norm(vec3.data(), vec3.size());
    ASSERT_NEAR(actual_norm3, expected_norm3, 1e-9);

    // Test case 4: Single element
    std::vector<double> vec4 = {7.0};
    double expected_norm4 = 7.0;
    double actual_norm4 = euclidean_vec_norm(vec4.data(), vec4.size());
    ASSERT_NEAR(actual_norm4, expected_norm4, 1e-9);

    // Test case 5: Empty vector (should handle gracefully, e.g., return 0)
    std::vector<double> vec5;
    double expected_norm5 = 0.0;
    double actual_norm5 = euclidean_vec_norm(vec5.data(), vec5.size());
    ASSERT_NEAR(actual_norm5, expected_norm5, 1e-9);

    std::cout << "    All euclidean_vec_norm tests passed." << std::endl;
}

// Register the test cases in an anonymous namespace to prevent symbol conflicts
namespace {
    struct RegisterUtilitiesTests {
        RegisterUtilitiesTests() {
            register_test("Utilities::EuclideanVecNorm", test_euclidean_vec_norm);
            // Add other utility tests here as you create them
        }
    };
    // Create a static instance of the struct to ensure tests are registered
    static RegisterUtilitiesTests reg_utilities_tests;
} // namespace