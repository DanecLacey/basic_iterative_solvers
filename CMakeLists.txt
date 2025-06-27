cmake_minimum_required(VERSION 3.22.1)
project(basic_iterative_solvers CXX)

set(CPP_VERSION c++17 CACHE STRING "C++ version")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ version: ${CPP_VERSION}")

# ========= User Configuration Options =========

# Set default build type if not defined
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Provide options for build type
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Solver parameters
set(_MAX_ITERS 1000 CACHE STRING "")
set(_TOL 1e-14 CACHE STRING "")
set(_GMRES_RESTART_LEN 10 CACHE STRING "")
set(_RES_CHECK_LEN 1 CACHE STRING "")
set(_PRECOND_ITERS 1 CACHE STRING "")
set(_PRECOND_INNER_ITERS 1 CACHE STRING "")
set(_INIT_X_VAL 0.1 CACHE STRING "")
set(_B_VAL 1.0 CACHE STRING "")

# Debugging
option(DEBUG_MODE "Enable debug mode" OFF)
option(DEBUG_MODE_FINE "Enable fine-grained debug mode" OFF)
option(OUTPUT_SPARSITY "Enable sparsity output" OFF)

# External libraries
option(USE_LIKWID "Enable LIKWID performance monitoring" OFF)
option(USE_SCAMAC "Enable SCAMAC matrix generator" OFF)
option(USE_SMAX "Enable SMAX support" OFF)

# ========== Sources and Targets ==========


add_executable(basic_iterative_solvers
    main.cpp
    utilities/mmio.cpp
)

# ========== Tests ==========
add_executable(run_tests
    tests/test_main.cpp
    tests/test_utilities.cpp
    # Add more test files here as they are created.
    tests/test_kernels.cpp
    tests/test_solvers.cpp
    
    utilities/mmio.cpp
)

# NOTE: OpenMP always required
# ========= OpenMP Support =========
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    # Link both executables to OpenMP
    target_link_libraries(basic_iterative_solvers PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(run_tests PUBLIC OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found, but required.")
endif()

# === LIKWID Support ===
if(USE_LIKWID)
    find_package(likwid REQUIRED)
    if(likwid_FOUND)
        message(STATUS "LIKWID found, enabling LIKWID support.")
        add_compile_definitions(USE_LIKWID LIKWID_PERFMON)
        include_directories(${LIKWID_INCLUDES})
        set(LIKWID_LIBRARIES_TO_LINK ${LIKWID_LIBRARIES})
        # Link both executables to LIKWID
        target_link_libraries(basic_iterative_solvers PUBLIC ${LIKWID_LIBRARIES_TO_LINK})
        target_link_libraries(run_tests PUBLIC ${LIKWID_LIBRARIES_TO_LINK})
    else()
        message(FATAL_ERROR "LIKWID requested but not found.")
    endif()
else()
    message(STATUS "LIKWID support is disabled.")
endif()

# === SCAMAC Support ===
if(USE_SCAMAC)
    # To maintain compatibility with other package's file paths
    set(scamac_DIR ${CMAKE_PREFIX_PATH}/scamac/lib/cmake/)
    find_package(scamac REQUIRED)

    if(TARGET scamac::scamac)
        message(STATUS "SCAMAC target scamac::scamac found, enabling support.")
        # Link both executables to SCAMAC
        target_link_libraries(basic_iterative_solvers PUBLIC scamac::scamac)
        target_link_libraries(run_tests PUBLIC scamac::scamac)
        target_compile_definitions(basic_iterative_solvers PUBLIC USE_SCAMAC)
        target_compile_definitions(run_tests PUBLIC USE_SCAMAC) # Also for tests if needed
    else()
        message(FATAL_ERROR "SCAMAC target scamac::scamac not found.")
    endif()
else()
    message(STATUS "SCAMAC support is disabled.")
endif()

# === SMAX Support ===
if(USE_SMAX)
    find_package(SmaxKernels REQUIRED)
    if(SmaxKernels_FOUND)
        message(STATUS "SMAX found, enabling SMAX support.")
        # Link both executables to SMAX
        target_link_libraries(basic_iterative_solvers PRIVATE SMAX::smaxkernels)
        target_link_libraries(run_tests PRIVATE SMAX::smaxkernels)
        target_compile_definitions(basic_iterative_solvers PRIVATE USE_SMAX)
        target_compile_definitions(run_tests PRIVATE USE_SMAX) # Also for tests if needed
        # Set permutation mode (and set default if not defined)
        if(NOT SMAX_PERM_MODE)
            set(SMAX_PERM_MODE NONE CACHE STRING "Choose the permutation mode." FORCE)
        endif()
        set_property(CACHE SMAX_PERM_MODE PROPERTY STRINGS "NONE" "JH" "BFS")
        message(STATUS "Permutation Mode: ${SMAX_PERM_MODE}")
    else()
        message(FATAL_ERROR "SMAX requested but not found.")
    endif()
else()
    message(STATUS "SMAX support is disabled.")
endif()

# Set include directories for both main and test executables
# This tells the compiler where to find your .hpp files (headers)
target_include_directories(basic_iterative_solvers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} # Project root (for common.hpp, kernels.hpp, solver.hpp etc.)
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities
    ${CMAKE_CURRENT_SOURCE_DIR}/methods
    
)

target_include_directories(run_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} # Project root
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities
    ${CMAKE_CURRENT_SOURCE_DIR}/methods
    
)

# Optional: Add a custom command to easily run tests
add_custom_target(test_all COMMAND run_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# ========== Set Compiler and Flags ==========
# If the user didn't specify a compiler, default to g++
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "g++" CACHE STRING "Default compiler" FORCE)
    message(STATUS "No compiler selected, using default: ${CMAKE_CXX_COMPILER}")
endif()

# Display the selected compiler
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

# Normalize compiler ID for consistent matching
string(TOUPPER "${CMAKE_CXX_COMPILER_ID}" COMPILER_ID)

# Compiler-specific flags - apply to both targets
# Note: It's generally better to use target_compile_options for flags specific to a target
# but for global flags, CMAKE_CXX_FLAGS can be used.
if(COMPILER_ID STREQUAL "GNU")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fstrict-aliasing -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(COMPILER_ID STREQUAL "INTEL")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fno-alias -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
elseif(COMPILER_ID STREQUAL "INTELLLVM")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer -fp-model=precise")
    endif()
elseif(COMPILER_ID STREQUAL "CLANG")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-alias -O3")
    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_MODE -g -O0 -Wall -Wextra -fno-omit-frame-pointer")
    endif()
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()


# ========== Preprocessor Definitions ==========
add_definitions(
    -DMAX_ITERS=${_MAX_ITERS}
    -DTOL=${_TOL}
    -DGMRES_RESTART_LEN=${_GMRES_RESTART_LEN}
    -DRES_CHECK_LEN=${_RES_CHECK_LEN}
    -DINIT_X_VAL=${_INIT_X_VAL}
    -DB_VAL=${_B_VAL}
    -DPRECOND_ITERS=${_PRECOND_ITERS}
    -DPRECOND_INNER_ITERS=${_PRECOND_INNER_ITERS}
)

if(DEBUG_MODE)
    add_definitions(-DDEBUG_MODE)
    add_compile_options(-g)
endif()

if(DEBUG_MODE_FINE)
    add_definitions(-DDEBUG_MODE_FINE)
endif()

if(OUTPUT_SPARSITY)
    add_definitions(-DOUTPUT_SPARSITY)
endif()

if(USE_SMAX)
    add_definitions(-DPERM_MODE=${SMAX_PERM_MODE})
endif()
