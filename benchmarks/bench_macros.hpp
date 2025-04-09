#ifndef BENCH_MACROS_HPP
#define BENCH_MACROS_HPP

#define MIN_BENCH_TIME 1.0
#define MIN_NUM_ITERS 100
#define SPLTSV_FLOPS_PER_NZ 2
#define SPLTSV_FLOPS_PER_ROW 2
#define GF_TO_F 1000000000
#define F_TO_GF 0.000000001
#define TOL 1e-8

#ifndef INIT_X_VAL
#define INIT_X_VAL 0
#endif

#ifndef B_VAL
#define B_VAL 1
#endif 

#ifdef USE_SCAMAC
    #define READ(coo_mat, cli_args)                                 \
    {                                                               \
            IF_DEBUG_MODE(printf("Generating SCAMAC Matrix\n"));    \
            coo_mat->scamac_make_mtx(cli_args->matrix_file_name);   \
    }
#else
    #define READ(coo_mat, cli_args)                                 \
    {                                                               \
            IF_DEBUG_MODE(printf("Reading .mtx Matrix\n"));         \
            coo_mat->read_from_mtx(cli_args->matrix_file_name);     \
    }
#endif

#define ALLOC_INIT(D, x, b, crs_mat)                \
{                                                   \
    D = new double[(crs_mat)->n_cols];              \
    x = new double[(crs_mat)->n_cols];              \
    b = new double[(crs_mat)->n_cols];              \
                                                    \
    _Pragma("omp parallel for")                     \
    for (int i = 0; i < (crs_mat)->n_cols; ++i) {   \
        x[i] = INIT_X_VAL;                          \
        b[i] = B_VAL;                               \
        D[i] = 0.0;                                 \
    }                                               \
}

#define PRINT_BENCH(bench_name, cli_args, n_threads, runtime, n_iter, crs_mat_L)   \
{                                                                                           \
    std::cout << "----------------" << std::endl;                                           \
    std::cout << "--" << bench_name << " Bench--" << std::endl;                                           \
    std::cout << (cli_args)->matrix_file_name << " with " << (n_threads)                    \
                << " thread(s)" << std::endl;                                               \
    std::cout << "Runtime: " << (runtime) << std::endl;                              \
    std::cout << "Iterations: " << (n_iter) << std::endl;                            \
                                                                                            \
    long flops_per_iter = ((crs_mat_L)->nnz * SPLTSV_FLOPS_PER_NZ +                                \
                            (crs_mat_L)->n_rows * SPLTSV_FLOPS_PER_ROW);                           \
    long iter_per_second = static_cast<long>((n_iter) / (runtime));           \
                                                                                            \
    std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF              \
                << " [GF/s]" << std::endl;                                                  \
    std::cout << "----------------" << std::endl;                                           \
} 

#define CLEANUP(bench_harness)         \
{                                          \
    delete cli_args;                       \
    delete coo_mat;                        \
    delete crs_mat;                        \
    delete coo_mat_L;                      \
    delete coo_mat_U;                      \
    delete crs_mat_L;                      \
    delete (bench_harness);                \
    delete[] D;                            \
    delete[] x;                            \
    delete[] b;                            \
}

#endif
