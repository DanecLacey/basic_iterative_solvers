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
set(_PRECOND_ITERS 0 CACHE STRING "")
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

# NOTE: OpenMP always required
# ========= OpenMP Support =========
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    target_link_libraries(basic_iterative_solvers PUBLIC OpenMP::OpenMP_CXX)
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
        target_link_libraries(basic_iterative_solvers PUBLIC ${LIKWID_LIBRARIES_TO_LINK})
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
        target_link_libraries(basic_iterative_solvers PUBLIC scamac::scamac)
        target_compile_definitions(basic_iterative_solvers PUBLIC USE_SCAMAC)
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
        target_link_libraries(basic_iterative_solvers PRIVATE SMAX::smaxkernels)
        target_compile_definitions(basic_iterative_solvers PRIVATE USE_SMAX)
    else()
        message(FATAL_ERROR "SMAX requested but not found.")
    endif()
else()
    message(STATUS "SMAX support is disabled.")
endif()

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

# Compiler-specific flags
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