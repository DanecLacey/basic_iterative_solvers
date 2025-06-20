#ifndef TEST_FRAMEWORK_HPP
#define TEST_FRAMEWORK_HPP

#include <vector>
#include <string>
#include <functional> 

// Struct to hold test case information
struct TestCase 
{
    std::string name;
    std::function<void()> func;
};

// Declare the global list of test cases as extern
// The definition will be in test_main.cpp
extern std::vector<TestCase> g_test_cases;

// Declare the function to register tests
void register_test(const std::string& name, std::function<void()> func);

// Declare the function to run all tests
void run_all_tests();

#endif // TEST_FRAMEWORK_HPP