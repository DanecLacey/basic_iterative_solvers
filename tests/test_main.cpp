// This file will contain the main function for running all tests.
// It provides the definitions for the test framework.
#include "test_framework.hpp" // Include the new header

#include <iostream> // For std::cout, std::cerr
#include <stdexcept> // For std::runtime_error

// Define the global list to store all test cases
std::vector<TestCase> g_test_cases; // No 'extern' here, this is the definition

// Define the function to register tests
void register_test(const std::string& name, std::function<void()> func) {
    g_test_cases.push_back({name, func});
}

// Define the function to run all tests
void run_all_tests() {
    std::cout << "--- Running All Tests ---" << std::endl;
    int passed_count = 0;
    for (const auto& test_case : g_test_cases) {
        std::cout << "[RUNNING] " << test_case.name << std::endl;
        try {
            test_case.func();
            std::cout << "[PASSED] " << test_case.name << std::endl;
            passed_count++;
        } catch (const std::exception& e) {
            std::cerr << "[FAILED] " << test_case.name << ": " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[FAILED] " << test_case.name << ": Unknown error" << std::endl;
        }
    }
    std::cout << "\n--- Test Summary ---" << std::endl;
    std::cout << "Total tests: " << g_test_cases.size() << std::endl;
    std::cout << "Passed: " << passed_count << std::endl;
    std::cout << "Failed: " << g_test_cases.size() - passed_count << std::endl;
    if (passed_count == g_test_cases.size()) {
        std::cout << "All tests passed successfully!" << std::endl;
    } else {
        std::cout << "Some tests failed." << std::endl;
    }
}

// The main function of the test executable
int main() {
    run_all_tests();
    return 0;
}