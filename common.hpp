#ifndef COMMON_HPP
#define COMMON_HPP

#include <sys/time.h>
#include <string>
#include <iostream>

class Stopwatch
{

long double wtime{};

public:
	timeval *begin;
	timeval *end;
	Stopwatch(timeval *_begin, timeval *_end) : begin(_begin), end(_end) {};
	Stopwatch() : begin(), end() {};

	void start(void)
	{
			gettimeofday(begin, 0);
	}

	void stop(void)
	{
			gettimeofday(end, 0);
			long seconds = end->tv_sec - begin->tv_sec;
			long microseconds = end->tv_usec - begin->tv_usec;
			wtime += seconds + microseconds*1e-6;
	}

	long double check(void)
	{
			gettimeofday(end, 0);
			long seconds = end->tv_sec - begin->tv_sec;
			long microseconds = end->tv_usec - begin->tv_usec;
			return seconds + microseconds*1e-6;
	}

	long double get_wtime(
	){
			return wtime;
	}
};

#define CREATE_STOPWATCH(timer_name) \
    timeval *timer_name##_time_start = new timeval; \
    timeval *timer_name##_time_end = new timeval; \
    Stopwatch *timer_name##_time = new Stopwatch(timer_name##_time_start, timer_name##_time_end); \
    timers->timer_name##_time = timer_name##_time;

#define TIME(timer_name, routine) \
		timer_name##_time->start(); \
		routine; \
		timer_name##_time->stop();

#ifdef DEBUG_MODE
    #define IF_DEBUG_MODE(print_statement) print_statement;
#else
    #define IF_DEBUG_MODE(print_statement)
#endif

struct Timers
{
	Stopwatch *total_time;
	Stopwatch *preprocessing_time;
	Stopwatch *solve_time;
	Stopwatch *per_iteration_time;
	Stopwatch *iterate_time;
	Stopwatch *sample_time;
	Stopwatch *exchange_time;
	Stopwatch *save_x_star_time;
	Stopwatch *postprocessing_time;
};

struct Args {
		std::string matrix_file_name{};
    std::string solver_type{};
    std::string preconditioner_type{};
};

class SanityChecker {
public:
	template<typename VT>
	static void print_vector(VT *vector, int size, std::string vector_name){
		std::cout << vector_name << ": [" << std::endl;
		for(int i = 0; i < size; ++i){
			std::cout << vector[i] << ", ";
		}
		std::cout << std::endl;
	}
};

#endif