# [gcc, icc, icx, clang]
COMPILER = gcc
# [int]
SIMD_LENGTH = 8
# [c++14]
CPP_VERSION = c++14

### Solver Parameters ###
# [int]
MAX_ITERS=5000
# [float]
TOL=1e-14
# [int]
GMRES_RESTART_LEN=50
# [int]
RES_CHECK_LEN=1
# [int]
PRECOND_ITERS=1
# [float]
INIT_X_VAL=0.1
# [float]
B_VAL=1.0

### Debugging ###
 # [1/0]
DEBUG_MODE = 0
 # [1/0]
DEBUG_MODE_FINE = 0
 # [1/0]
OUTPUT_SPARSITY = 0

### External Libraries ###
# [1/0]
USE_LIKWID = 0
# LIKWID_INC =
# LIKWID_LIB = 

# [1/0]
USE_SCAMAC = 0
# SCAMAC_INC = 
# SCAMAC_LIB = 