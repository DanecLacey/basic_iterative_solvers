#ifndef KERNEL_BENCHMARKS_HPP
#define KERNEL_BENCHMARKS_HPP

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

int kernel_benchmarks(int argc, char *argv[]){

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

	// SpLTSV Bench
	double spltsv_runtime = 0.0;
	int n_spltsv_iter = MIN_NUM_ITERS;
	int n_threads = 1;
#ifdef _OPENMP
	#pragma omp parallel
	{
	n_threads = omp_get_num_threads();
	}
#endif

#ifdef USE_LIKWID
	LIKWID_MARKER_REGISTER("spltsv");
#endif

	// // TODO: JH
	// // Select spltsv variant to bench
	// std::function<void(bool)> lambda;
	// std::string bench_name;

	// if (cli_args->exp_kernels["spltsv"] == "lvl") {
	// 	lambda = [crs_mat_L, x, D, b](bool warmup) { spltsv_lvl(crs_mat_L, x, D, b, warmup); };
	// 	bench_name = "spltsv_lvl";
	// } 
	// else if (cli_args->exp_kernels["spltsv"] == "2gs") {
	// 	lambda = [crs_mat_L, x, D, b](bool warmup) { spltsv_2stage(crs_mat_L, x, D, b, warmup); };
	// 	bench_name = "spltsv_2stage";
	// } 
	// else if (cli_args->exp_kernels["spltsv"] == "2mc") {
	// 	lambda = [crs_mat_L, x, D, b](bool warmup) { spltsv_2mc(crs_mat_L, x, D, b, warmup); };
	// 	bench_name = "spltsv_2mc";
	// } 
	// else {
	// 	lambda = [crs_mat_L, x, D, b](bool warmup) { spltsv(crs_mat_L, x, D, b, warmup); };
	// 	bench_name = "spltsv";
	// }

	std::function<void(bool)> lambda = [crs_mat_L, x, D, b](bool warmup) { spltsv(crs_mat_L, x, D, b, warmup); };
	std::string bench_name = "spltsv";

	BenchHarness *spltsv_bench_harness = new BenchHarness(
		bench_name,
		lambda,
		n_spltsv_iter,
		spltsv_runtime, 
		MIN_BENCH_TIME
	);

	IF_DEBUG_MODE(printf("Running SpLTSV bench\n"))
	spltsv_bench_harness->warmup(true);
	IF_DEBUG_MODE(printf("Warmup complete\n"))

	spltsv_bench_harness->bench(false);
	IF_DEBUG_MODE(printf("Bench complete\n"))

	// Report results
    PRINT_BENCH("SpLTSV", cli_args, n_threads, spltsv_runtime, n_spltsv_iter, crs_mat_L);

    CLEANUP(spltsv_bench_harness);

	return 0;
}

#endif
