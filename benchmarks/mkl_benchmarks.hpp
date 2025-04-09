#ifndef MKL_BENCHMARKS_HPP
#define MKL_BENCHMARKS_HPP

#include "bench_harness.hpp"
#include "../kernels.hpp"
#include "../common.hpp"
#include "../utilities.hpp"
#include "bench_macros.hpp"

#ifdef USE_LIKWID
	#include <likwid-marker.h>
#endif

#ifdef _OPENMP
	#include <omp.h>
#endif

#ifdef BENCH_MKL
    #include <mkl.h>
#endif

int mkl_benchmarks(int argc, char *argv[]){
#ifdef BENCH_MKL
	Args *cli_args = new Args;
	parse_cli(cli_args, argc, argv, true);

	MatrixCOO *coo_mat = new MatrixCOO;
    READ(coo_mat, cli_args);

	IF_DEBUG_MODE(printf("Converting COO matrix to CRS\n"))
	MatrixCRS *crs_mat = new MatrixCRS;
	convert_coo_to_crs(coo_mat, crs_mat);

	double *D; double *x; double *b;
    ALLOC_INIT(D, x, b, crs_mat);

	IF_DEBUG_MODE(printf("Extracting L, U, and D\n"))
	MatrixCOO *coo_mat_L = new MatrixCOO;
	MatrixCOO *coo_mat_U = new MatrixCOO;
	extract_L_U(coo_mat, coo_mat_L, coo_mat_U);
	extract_D(coo_mat, D);

	IF_DEBUG_MODE(printf("Converting L to CRS\n"))
	MatrixCRS *crs_mat_L = new MatrixCRS;
	convert_coo_to_crs(coo_mat_L, crs_mat_L);


    // Create matrix handle
    sparse_matrix_t L;
    mkl_sparse_d_create_csr(
        &L, 
        SPARSE_INDEX_BASE_ZERO,
        crs_mat_L->n_rows,
        crs_mat_L->n_cols, 
        crs_mat_L->row_ptr,
        crs_mat_L->row_ptr + 1,
        crs_mat_L->col,
        crs_mat_L->val
    );

    // Set matrix descriptor for lower triangular
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    descr.mode = SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    // MKL Sparse TRSV Bench
	double mkl_trsv_runtime = 0.0;
	int n_mkl_trsv_iter = MIN_NUM_ITERS;
	int n_threads = 1;
#ifdef _OPENMP
	#pragma omp parallel
	{
	n_threads = omp_get_num_threads();
	}
#endif

    // NOTE: I don't know if this works as expected
    mkl_set_num_threads(n_threads); 

#ifdef USE_LIKWID
	LIKWID_MARKER_REGISTER("mkl_trsv");
#endif

    BenchHarness *mkl_trsv_bench_harness = new BenchHarness(
        "mkl_trsv",
        [L, descr, b, x](bool warmup){
#ifdef USE_LIKWID
            if(!warmup){
                LIKWID_MARKER_START("mkl_trsv");
            }
#endif
            mkl_sparse_d_trsv(
                SPARSE_OPERATION_NON_TRANSPOSE, 1.0, L, descr, b, x
            );
#ifdef USE_LIKWID
            if(!warmup){
                LIKWID_MARKER_STOP("mkl_trsv");
            }
#endif
        },
        n_mkl_trsv_iter,
        mkl_trsv_runtime, 
        MIN_BENCH_TIME
    );

    IF_DEBUG_MODE(printf("Running MKL TRSV bench\n"))
    mkl_trsv_bench_harness->warmup(true);
    IF_DEBUG_MODE(printf("Warmup complete\n"))
    mkl_trsv_bench_harness->bench(false);
    IF_DEBUG_MODE(printf("Bench complete\n"))

	// Report results

    PRINT_BENCH("MKL TRSV", cli_args, n_threads, mkl_trsv_runtime, n_mkl_trsv_iter, crs_mat_L);

    mkl_sparse_destroy(L);
    CLEANUP(mkl_trsv_bench_harness);

    return 0;
#else
    printf("MKL is not enabled. Please compile with -DBENCH_MKL to enable MKL benchmarks.\n");
    return 1;
#endif
}

#endif
