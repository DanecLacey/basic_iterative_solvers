#ifndef BENCH_HARNESS_HPP
#define BENCH_HARNESS_HPP

#include "timing.h"

#include <functional>
#include <iostream>

class BenchHarness {
public:
	const std::string& bench_name;
	std::function<void(bool)> callback;
	int &n_iters;
	double &runtime;
	double min_bench_time;
  int *counter;

	BenchHarness(const std::string& _bench_name, std::function<void(bool)> _callback, int &_n_iters, double &_runtime, double _min_bench_time, int *_counter = nullptr)
  : bench_name(_bench_name), callback(_callback), n_iters(_n_iters), runtime(_runtime), min_bench_time(_min_bench_time), counter(_counter)
	{};
	// ~BenchHarness();

	void warmup(){
		double warmup_begin_loop_time, warmup_end_loop_time;
    warmup_begin_loop_time = warmup_end_loop_time = 0.0;

		int warmup_n_iters = n_iters;
		double warmup_runtime = 0.0;

    do{
        warmup_begin_loop_time = getTimeStamp();
        for(int k = 0; k < warmup_n_iters; ++k) {
            callback(true);
#ifdef DEBUG_MODE_FINE
            std::cout << "Completed warmup_" << bench_name << " iter " << k << std::endl;
#endif
        }
        if(counter != nullptr) (*counter) += warmup_n_iters;
        warmup_n_iters *= 2;
        warmup_end_loop_time = getTimeStamp();
        warmup_runtime = warmup_end_loop_time - warmup_begin_loop_time;
    } while (warmup_runtime < min_bench_time);
    warmup_n_iters /= 2;
		
	};

	void bench(){
		double begin_loop_time, end_loop_time;
    begin_loop_time = end_loop_time = 0.0;

    do{
        begin_loop_time = getTimeStamp();
        for(int k = 0; k < n_iters; ++k) {
            callback(false);
#ifdef DEBUG_MODE_FINE
            std::cout << "Completed " << bench_name << " iter " << k << std::endl;
#endif
        }
        if(counter != nullptr) (*counter) += n_iters;
        n_iters *= 2;
        end_loop_time = getTimeStamp();
        runtime = end_loop_time - begin_loop_time;
    } while (runtime < min_bench_time);
    n_iters /= 2;
		
	};

};

#endif // BENCH_HARNESS_HPP
