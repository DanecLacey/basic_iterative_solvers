cmake_minimum_required(VERSION 3.15)
project(BasicIterativeSolvers CXX)

# ========= User Configuration Options =========
# Set default values (you can override these via -D options when calling CMake)
set(COMPILER "gcc" CACHE STRING "Compiler to use [gcc, icc, icx, clang]")
set(CPP_VERSION c++14 CACHE STRING "C++ version")

# Solver parameters
set(_MAX_ITERS 1000 CACHE STRING "")
set(_TOL 1e-14 CACHE STRING "")
set(_GMRES_RESTART_LEN 10 CACHE STRING "")
set(_RES_CHECK_LEN 1 CACHE STRING "")
set(_PRECOND_ITERS 1 CACHE STRING "")
set(_INIT_X_VAL 0.1 CACHE STRING "")
set(_B_VAL 1.0 CACHE STRING "")

# Debugging
option(DEBUG_MODE "Enable debug mode" OFF)
option(DEBUG_MODE_FINE "Enable fine-grained debug mode" OFF)
option(OUTPUT_SPARSITY "Enable sparsity output" OFF)

# External libraries
option(USE_SCAMAC "Use SCAMAC matrix generator" OFF)
option(USE_MKL "Enable MKL support" OFF)
option(USE_SMAX "Enable SMAX support" OFF)

# === SCAMAC Support ===
# TODO
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(USE_SCAMAC)
    if(NOT EXISTS ${SCAMAC_INC})
        message(FATAL_ERROR "SCAMAC_INC not found: ${SCAMAC_INC}")
    endif()
    if(NOT EXISTS ${SCAMAC_LIB})
        message(FATAL_ERROR "SCAMAC_LIB not found: ${SCAMAC_LIB}")
    endif()
    include_directories(${SCAMAC_INC})
    add_definitions(-DUSE_SCAMAC)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${SCAMAC_LIB})
endif()

# === MKL Support ===
if(USE_MKL)
    # Ensure we're using LP64 interface (32-bit integers) for MKL
    # TODO: This isn't explicitly necessary. Make more flexible.
    set(MKL_INTERFACE lp64)
    find_package(MKL REQUIRED)
    if(MKL_FOUND)
        message(STATUS "MKL found, enabling MKL support.")
    else()
        message(FATAL_ERROR "MKL requested but not found.")
    endif()
else()
    message(STATUS "MKL support is disabled.")
endif()

# === SMAX Support ===
if(USE_SMAX)
    find_package(SmaxKernels REQUIRED)
    if(SmaxKernels_FOUND)
        message(STATUS "SMAX found, enabling SMAX support.")
    else()
        message(FATAL_ERROR "SMAX requested but not found.")
    endif()
else()
    message(STATUS "SMAX support is disabled.")
endif()

# ========== Set Compiler and Flags ==========
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(COMPILER STREQUAL "gcc")
    set(CMAKE_CXX_COMPILER g++)
    add_compile_options(-O3 -Wall -fopenmp -march=native)
elseif(COMPILER STREQUAL "icc")
    set(CMAKE_CXX_COMPILER icpc)
    add_compile_options(-O3 -Wall -fopenmp -xhost)
elseif(COMPILER STREQUAL "icx")
    set(CMAKE_CXX_COMPILER icpx)
    add_compile_options(-O3 -Wall -fopenmp -xhost -Xclang -target-feature -Xclang +prefer-no-gather)
elseif(COMPILER STREQUAL "clang")
    set(CMAKE_CXX_COMPILER clang++)
    add_compile_options(-Ofast -Wall -fopenmp -march=native)
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

# ========== Sources and Targets ==========
add_executable(basic_iterative_solvers
    main.cpp
    mmio.cpp
)

target_link_libraries(basic_iterative_solvers ${EXTERNAL_LIBS})