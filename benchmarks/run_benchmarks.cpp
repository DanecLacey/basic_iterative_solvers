#include "kernel_benchmarks.hpp"
#include "mkl_benchmarks.hpp"
#include "validate.hpp"

int main(int argc, char *argv[]){
#ifdef USE_LIKWID
	LIKWID_MARKER_INIT;
#endif

	kernel_benchmarks(argc, argv);

	// Only bench MKL if the results agree with ours
#ifdef BENCH_MKL
	if(!validate(argc, argv))
		mkl_benchmarks(argc, argv);
#endif

#ifdef USE_LIKWID
	LIKWID_MARKER_CLOSE;
#endif

	return 0;
}
