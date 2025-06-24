#ifndef COMMON_HPP
#define COMMON_HPP

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <sys/time.h>
#include <unordered_map>

#ifdef USE_SMAX
#include "SmaxKernels/interface.hpp"
#endif

#ifdef USE_SMAX
using Interface = SMAX::Interface;
#define SMAX_ARGS(...) , __VA_ARGS__
#else
using Interface = void *;
#define SMAX_ARGS(...)
#endif

#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif

#ifndef STRINGIFY
#define STRINGIFY(x) #x
#endif

#ifndef TO_STRING
#define TO_STRING(x) STRINGIFY(x)
#endif

enum class PrecondType {
    None,
    Jacobi,
    GaussSeidel,
    BackwardsGaussSeidel,
    SymmetricGaussSeidel,
    TwoStageGS
};

enum class SolverType {
    Jacobi,
    GaussSeidel,
    SymmetricGaussSeidel,
    GMRES,
    ConjugateGradient,
    BiCGSTAB
};

// Primary template (undefined to cause a compile error if not specialized)
template <typename EnumType> std::string to_string(EnumType);

// PrecondType specialization
template <> inline std::string to_string(PrecondType type) {
    switch (type) {
    case PrecondType::Jacobi:
        return "jacobi";
    case PrecondType::GaussSeidel:
        return "gauss-seidel";
    case PrecondType::BackwardsGaussSeidel:
        return "backwards-gauss-seidel";
    case PrecondType::SymmetricGaussSeidel:
        return "symmetric-gauss-seidel";
    case PrecondType::TwoStageGS:
        return "two-stage-gauss-seidel";
    case PrecondType::None:
        return "none";
    default:
        return "unknown";
    }
}

// SolverType specialization
template <> inline std::string to_string(SolverType type) {
    switch (type) {
    case SolverType::Jacobi:
        return "jacobi";
    case SolverType::GaussSeidel:
        return "gauss-seidel";
    case SolverType::SymmetricGaussSeidel:
        return "symmetric-gauss-seidel";
    case SolverType::GMRES:
        return "gmres";
    case SolverType::ConjugateGradient:
        return "conjugate-gradient";
    case SolverType::BiCGSTAB:
        return "bicgstab";
    default:
        return "unknown";
    }
}

struct Args {
    std::string matrix_file_name;
    SolverType method;
    PrecondType preconditioner;
};

void *aligned_malloc(size_t bytesize) {
    int errorCode;
    void *ptr;

    errorCode = posix_memalign(&ptr, ALIGNMENT, bytesize);

    if (errorCode) {
        if (errorCode == EINVAL) {
            fprintf(stderr,
                    "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM) {
            fprintf(stderr,
                    "Error: Insufficient memory to fulfill the request\n");
            exit(EXIT_FAILURE);
        }
    }

    if (ptr == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

// Overload new and delete for alignement
void *operator new(size_t bytesize) {
    // printf("Overloading new operator with size: %lu\n", bytesize);
    int errorCode;
    void *ptr;
    errorCode = posix_memalign(&ptr, ALIGNMENT, bytesize);

    if (errorCode) {
        if (errorCode == EINVAL) {
            fprintf(stderr,
                    "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM) {
            fprintf(stderr, "Error: Insufficient memory to fulfill the request "
                            "for space\n");
            exit(EXIT_FAILURE);
        }
    }

    if (ptr == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

void operator delete(void *p) {
    // printf("Overloading delete operator\n");
    free(p);
}

class Stopwatch {

    long double wtime{};

  public:
    timeval *begin;
    timeval *end;
    Stopwatch(timeval *_begin, timeval *_end) : begin(_begin), end(_end) {};
    Stopwatch() : begin(), end() {};

    void start(void) { gettimeofday(begin, 0); }

    void stop(void) {
        gettimeofday(end, 0);
        long seconds = end->tv_sec - begin->tv_sec;
        long microseconds = end->tv_usec - begin->tv_usec;
        wtime += seconds + microseconds * 1e-6;
    }

    long double check(void) {
        gettimeofday(end, 0);
        long seconds = end->tv_sec - begin->tv_sec;
        long microseconds = end->tv_usec - begin->tv_usec;
        return seconds + microseconds * 1e-6;
    }

    long double get_wtime() { return wtime; }

    ~Stopwatch(){
        delete begin; delete end;
    }
};

#define CREATE_STOPWATCH(timer_name)                                           \
    timeval *timer_name##_time_start = new timeval;                            \
    timeval *timer_name##_time_end = new timeval;                              \
    Stopwatch *timer_name##_time =                                             \
        new Stopwatch(timer_name##_time_start, timer_name##_time_end);         \
    timers->timer_name##_time = timer_name##_time;

#define DELETE_STOPWATCH(timer_name)                                           \
    delete timer_name;               

#define TIME(timer_name, routine)                                              \
    do {                                                                       \
        timer_name##_time->start();                                            \
        routine;                                                               \
        timer_name##_time->stop();                                             \
    } while (0);

#ifdef DEBUG_MODE
#define IF_DEBUG_MODE(print_statement) print_statement;
#else
#define IF_DEBUG_MODE(print_statement)
#endif

#ifdef DEBUG_MODE_FINE
#define IF_DEBUG_MODE_FINE(print_statement) print_statement;
#else
#define IF_DEBUG_MODE_FINE(print_statement)
#endif

struct Timers {
    Stopwatch *total_time;
    Stopwatch *preprocessing_time;
    Stopwatch *solve_time;
    Stopwatch *per_iteration_time;
    Stopwatch *iterate_time;
    Stopwatch *spmv_time;
    Stopwatch *precond_time;
    Stopwatch *dgemm_time;
    Stopwatch *dgemv_time;
    Stopwatch *normalize_time;
    Stopwatch *dot_time;
    Stopwatch *sum_time;
    Stopwatch *copy1_time;
    Stopwatch *copy2_time;
    Stopwatch *norm_time;
    Stopwatch *scale_time;
    Stopwatch *sptrsv_time;
    Stopwatch *orthog_time;
    Stopwatch *least_sq_time;
    Stopwatch *update_g_time;
    Stopwatch *sample_time;
    Stopwatch *exchange_time;
    Stopwatch *restart_time;
    Stopwatch *save_x_star_time;
    Stopwatch *postprocessing_time;

    ~Timers() {
        DELETE_STOPWATCH(total_time);
        DELETE_STOPWATCH(preprocessing_time);
        DELETE_STOPWATCH(solve_time);
        DELETE_STOPWATCH(per_iteration_time);
        DELETE_STOPWATCH(iterate_time);
        DELETE_STOPWATCH(spmv_time);
        DELETE_STOPWATCH(precond_time);
        DELETE_STOPWATCH(dot_time);
        DELETE_STOPWATCH(copy1_time);
        DELETE_STOPWATCH(copy2_time);
        DELETE_STOPWATCH(normalize_time);
        DELETE_STOPWATCH(sum_time);
        DELETE_STOPWATCH(norm_time);
        DELETE_STOPWATCH(scale_time);
        DELETE_STOPWATCH(sptrsv_time);
        DELETE_STOPWATCH(dgemm_time);
        DELETE_STOPWATCH(dgemv_time);
        DELETE_STOPWATCH(orthog_time);
        DELETE_STOPWATCH(least_sq_time);
        DELETE_STOPWATCH(update_g_time);
        DELETE_STOPWATCH(sample_time);
        DELETE_STOPWATCH(exchange_time);
        DELETE_STOPWATCH(restart_time);
        DELETE_STOPWATCH(save_x_star_time);
        DELETE_STOPWATCH(postprocessing_time);
    }
};

class SanityChecker {
  public:
    template <typename VT>
    static void print_vector(VT *vector, int size, std::string vector_name) {
        std::cout << vector_name << " : [" << std::endl;
        for (int i = 0; i < size; ++i) {
            std::cout << vector[i] << ", ";
        }
        std::cout << "]" << std::endl;
    }

    template <typename VT>
    static void print_dense_mat(VT *A, int n_rows, int n_cols,
                                std::string mat_name) {
        int fixed_width = 12;
        std::cout << mat_name << ": [" << std::endl;
        for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
            for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
                std::cout << std::setw(fixed_width);
                std::cout << A[(n_cols * row_idx) + col_idx] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    static void print_extract_L_U_error(int nz_idx) {
        fprintf(stderr, "ERROR: extract_L_U: nz_idx %i cannot be segmented.\n",
                nz_idx);
        exit(EXIT_FAILURE);
    }

    static void print_extract_L_plus_D(int nz_idx) {
        fprintf(
            stderr,
            "ERROR: print_extract_L_plus_D: nz_idx %i cannot be segmented.\n",
            nz_idx);
        exit(EXIT_FAILURE);
    }

    static void zero_diag(int row_idx) {
        fprintf(stderr, "Zero detected on diagonal at row index %d\n", row_idx);
        exit(EXIT_FAILURE);
    }

    static void no_diag(int row_idx) {
        fprintf(stderr, "No diagonal to extract at row index %d\n", row_idx);
        exit(EXIT_FAILURE);
    }

    static void print_gmres_iter_counts(int iter_count, int restart_count) {
        printf("gmres solve iter_count = %i\n", iter_count);
        printf("gmres solve restart_count = %i\n", restart_count);
    }

    static void print_bicgstab_vectors(int N, double *x_new, double *x_old,
                                       double *tmp, double *p_new,
                                       double *p_old, double *residual_new,
                                       double *residual_old, double *residual_0,
                                       double *v, double *h, double *s,
                                       double *t, double rho_new,
                                       double rho_old, std::string phase) {
        std::cout << phase << std::endl;
        print_vector<double>(x_new, N, "x_new");
        print_vector<double>(x_old, N, "x_old");
        print_vector<double>(tmp, N, "tmp");
        print_vector<double>(p_new, N, "p_new");
        print_vector<double>(p_old, N, "p_old");
        print_vector<double>(residual_new, N, "residual_new");
        print_vector<double>(residual_old, N, "residual_old");
        print_vector<double>(residual_0, N, "residual_0");
        print_vector<double>(v, N, "v");
        print_vector<double>(h, N, "h");
        print_vector<double>(s, N, "s");
        print_vector<double>(t, N, "t");
        printf("rho_new = %f\n", rho_new);
        printf("rho_old = %f\n", rho_old);
    }

    static void check_V_orthonormal(double *V, int iter_count, int N) {
        // Check if all basis vectors in V are orthonormal
        double tol = 1e-14;

        // Computing euclidean norm
        for (int k = 0; k < iter_count + 1; ++k) {
            double tmp = 0.0;
            for (int i = 0; i < N; ++i) {
                tmp += V[k * N + i] * V[k * N + i];
            }
            double tmp_2_norm = std::sqrt(tmp);

            if (std::abs(tmp_2_norm) > 1 + tol) {
                printf(
                    "GMRES WARNING: basis vector v_%i has a norm of %.17g, \n \
								and does not have a norm of 1.0 as was expected.\n",
                    k, tmp_2_norm);
            } else {
                for (int j = iter_count; j > 0; --j) {
                    double tmp_dot;
                    // Takes new v_k, and compares with all other basis vectors
                    // in V

                    // Computing dot product
                    double sum = 0.0;
                    for (int i = 0; i < N; ++i) {
                        sum += V[(iter_count + 1) * N + i] * V[j * N + i];
                    }
                    tmp_dot = sum;

                    if (std::abs(tmp_dot) > tol) {
                        printf(
                            "GMRES WARNING: basis vector v_%i is not orthogonal to basis vector v_%i, \n \
										their dot product is %.17g, and not 0.0 as was expected.\n",
                            k, j, tmp_dot);
                    }
                }
            }
        }
    }

    static void check_H(double *H, double *R, double *Q, int restart_len) {
        // Validate that H == Q_tR [(m+1 x m) == (m+1 x m+1)(m+1 x m)]
        double tol = 1e-14;

        double *Q_t = new double[(restart_len + 1) * (restart_len + 1)];

// init
#pragma omp parallel for
        for (int i = 0; i < (restart_len + 1) * (restart_len + 1); ++i) {
            Q_t[i] = 0.0;
        }

        // transpose
        for (int row_idx = 0; row_idx < (restart_len + 1); ++row_idx) {
            for (int col_idx = 0; col_idx < (restart_len + 1); ++col_idx) {
                Q_t[col_idx * (restart_len + 1) + row_idx] =
                    Q[row_idx * (restart_len + 1) + col_idx];
            }
        }

        print_dense_mat<double>(Q_t, (restart_len + 1), (restart_len + 1),
                                "Q_t");

        double *Q_tR = new double[(restart_len + 1) * (restart_len)];

// init
#pragma omp parallel for
        for (int i = 0; i < (restart_len + 1) * restart_len; ++i) {
            Q_tR[i] = 0.0;
        }

        // Compute Q_tR <- Q_t*R [(m+1 x m) <- (m+1 x m+1)(m+1 x m)]
        for (int row_idx = 0; row_idx <= restart_len; ++row_idx) {
            for (int col_idx = 0; col_idx < restart_len; ++col_idx) {
                double sum = 0.0;
                for (int i = 0; i < (restart_len + 1); ++i) {
                    sum += Q_t[row_idx * (restart_len + 1) + i] *
                           R[col_idx + i * restart_len];
                }
                Q_tR[(row_idx * restart_len) + col_idx] = sum;
            }
        }

        print_dense_mat<double>(Q_tR, (restart_len + 1), restart_len, "Q_tR");

        // Scan and validate H=Q_tR
        for (int row_idx = 0; row_idx <= restart_len; ++row_idx) {
            for (int col_idx = 0; col_idx < restart_len; ++col_idx) {
                int idx = row_idx * restart_len + col_idx;
                if (std::abs(static_cast<double>(Q_tR[idx] - H[idx])) > tol) {
                    printf(
                        "GMRES WARNING: The Q_tR factorization of H at index %i has a value %.17g, \n \
										and does not have a value of %.17g as was expected.\n",
                        row_idx * restart_len + col_idx, Q_tR[idx],
                        H[row_idx * restart_len + col_idx]);
                }
            }
        }

        delete[] Q_t;
        delete[] Q_tR;
    }

    static void check_copied_L_U_elements(int total_nnz, int L_nnz, int U_nnz,
                                          int D_nnz) {
        int copied_elems_count = L_nnz + U_nnz + D_nnz;
        if (copied_elems_count != total_nnz) {
            fprintf(stderr,
                    "ERROR: extract_L_U: %i out of %i elements were copied "
                    "from coo_mat.\n",
                    copied_elems_count, total_nnz);
            exit(EXIT_FAILURE);
        }
    }

    static void check_copied_L_plus_D_elements(int total_nnz, int L_plus_D_nnz,
                                               int U_nnz) {
        int copied_elems_count = L_plus_D_nnz + U_nnz;
        if (copied_elems_count != total_nnz) {
            fprintf(stderr,
                    "ERROR: extract_L_plus_D: %i out of %i elements were "
                    "copied from coo_mat.\n",
                    copied_elems_count, total_nnz);
            exit(EXIT_FAILURE);
        }
    }
};

#endif
