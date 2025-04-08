#include "bench_harness.hpp"
#include "../kernels.hpp"
#include "../common.hpp"
#include "../utilities.hpp"

#ifdef USE_LIKWID
	#include <likwid-marker.h>
#endif


#ifdef _OPENMP
	#include <omp.h>
#endif

#define MIN_BENCH_TIME 1.0
#define MIN_NUM_ITERS 100
#define FLOPS_PER_NZ 2
#define FLOPS_PER_ROW 2
#define F_TO_GF 0.000000001

#ifndef INIT_X_VAL
#define INIT_X_VAL 0
#endif

#ifndef B_VAL
#define B_VAL 1
#endif 

int main(int argc, char *argv[]){

	Args *cli_args = new Args;
	parse_cli(cli_args, argc, argv, true);

	MatrixCOO *coo_mat = new MatrixCOO;
#ifdef USE_SCAMAC
	IF_DEBUG_MODE(printf("Generating SCAMAC Matrix\n"))
	coo_mat->scamac_make_mtx(cli_args->matrix_file_name);
#else
	IF_DEBUG_MODE(printf("Reading .mtx Matrix\n"))
	coo_mat->read_from_mtx(cli_args->matrix_file_name);
#endif

	IF_DEBUG_MODE(printf("Converting COO matrix to CRS\n"))
	MatrixCRS *crs_mat = new MatrixCRS;
	convert_coo_to_crs(coo_mat, crs_mat);

	double *D;
	double *x;
	double *b;
	D = new double [crs_mat->n_cols];
	x = new double [crs_mat->n_cols];
	b = new double [crs_mat->n_cols];

	#pragma omp parallel for
	for(int i = 0; i < crs_mat->n_cols; ++i){
		x[i] = INIT_X_VAL;
		b[i] = B_VAL;
		D[i] = 0.0;
	}

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

#ifdef USE_LIKWID
	LIKWID_MARKER_INIT;
	LIKWID_MARKER_REGISTER("spltsv");
#endif

	BenchHarness *spltsv_bench_harness = new BenchHarness(
		"spltsv",
		[crs_mat_L, x, D, b](bool warmup){ spltsv(crs_mat_L, x, D, b, warmup); },
		n_spltsv_iter,
		spltsv_runtime, 
		MIN_BENCH_TIME
	);

	IF_DEBUG_MODE(printf("Running SpLTSV bench\n"))
	spltsv_bench_harness->warmup();
	IF_DEBUG_MODE(printf("Warmup complete\n"))
	spltsv_bench_harness->bench();
	IF_DEBUG_MODE(printf("Bench complete\n"))


	// Report results
	int n_threads = 1;
#ifdef _OPENMP
	#pragma omp parallel
	{
	n_threads = omp_get_num_threads();
	}
#endif
	std::cout << "----------------" << std::endl;
	std::cout << "--SpLTSV Bench--" << std::endl;
	std::cout << cli_args->matrix_file_name << " with " << n_threads << " thread(s)" << std::endl;
	std::cout << "Runtime: " << spltsv_runtime << std::endl;
	std::cout << "Iterations: " << n_spltsv_iter << std::endl;

	int flops_per_iter = (crs_mat_L->nnz * FLOPS_PER_NZ + crs_mat_L->n_rows * FLOPS_PER_ROW);
	int iter_per_second = n_spltsv_iter / spltsv_runtime;
	std::cout << "Performance: " << flops_per_iter * iter_per_second * F_TO_GF << " [GF/s]" << std::endl;
	std::cout << "----------------" << std::endl;

	delete cli_args;
	delete coo_mat;
	delete crs_mat;
	delete coo_mat_L;
	delete coo_mat_U;
	delete crs_mat_L;
	delete spltsv_bench_harness;
	delete[] D;
	delete[] x;
	delete[] b;

#ifdef USE_LIKWID
	LIKWID_MARKER_CLOSE;
#endif

	return 0;
}