#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "common.hpp"
#include "sparse_matrix.hpp"
#include "kernels.hpp"
#include "methods/jacobi.hpp"
#include "methods/gauss_seidel.hpp"

#include <float.h>

class Solver
{
private:
	Args *cli_args;

public:
	MatrixCRS *crs_mat;
	MatrixCRS *crs_mat_L;
	MatrixCRS *crs_mat_U;
	std::string solver_type;
	std::string preconditioner_type;

	// Parameters
	int iter_count = 0;
	int collected_residual_norms_count = 0;
	double residual_norm = DBL_MAX;
	int max_iters;
	double tolerance;
	double stopping_criteria;
	int residual_check_len;

	// Flags
	bool convergence_flag = false;

	// Vectors
	double *x_star;
	double *x_new;
	double *x_old;
	double *x;
	double *b;
	double *D;
	double *tmp;
	double *residual;

	// Misc
	double *collected_residual_norms;
	double *time_per_iteration;

	Solver(Args *_cli_args) : cli_args(_cli_args) {
		solver_type = cli_args->solver_type;
		preconditioner_type = cli_args->preconditioner_type;

		// Assign config.mk parameters to the solver
		max_iters = MAX_ITERS;
		tolerance = TOL;
		residual_check_len = RES_CHECK_LEN;

		collected_residual_norms = new double[this->max_iters];
		time_per_iteration = new double[this->max_iters];
	}

	bool check_stopping_criteria(){
		bool norm_convergence = this->residual_norm < this->stopping_criteria;
		bool over_max_iters = this->iter_count >= this->max_iters;
		bool divergence = this->residual_norm > DBL_MAX;
		return norm_convergence || over_max_iters || divergence;
	}

	void apply_preconditioner(){

	}

	void init_structs(){
		double *x_star =   new double[this->crs_mat->n_cols];
		double *x_new =    new double[this->crs_mat->n_cols];
		double *x_old =    new double[this->crs_mat->n_cols];
		double *x =        new double[this->crs_mat->n_cols];
		double *b =        new double[this->crs_mat->n_cols];
		double *D =        new double[this->crs_mat->n_cols];
		double *tmp =      new double[this->crs_mat->n_cols];
		double *residual = new double[this->crs_mat->n_cols];

		#pragma omp parallel for
		for(int i = 0; i < this->crs_mat->n_cols; ++i){
			x_star[i] =   0.0;
			x_new[i] =    0.0;
			x_old[i] =    INIT_X_VAL;
			x[i] =        0.0;
			b[i] =        B_VAL;
			D[i] =        0.0;
			tmp[i] =      0.0;
			residual[i] = 0.0;
		}

		for(int i = 0; i < this->max_iters; ++i){
			collected_residual_norms[i] = 0.0;
			time_per_iteration[i] = 0.0;
		}

		this->x_star = x_star;
		this->x_new = x_new;
		this->x_old = x_old;
		this->x = x;
		this->b = b;
		this->D = D;
		this->tmp = tmp;
		this->residual = residual;

		this->collected_residual_norms = collected_residual_norms;
		this->time_per_iteration = time_per_iteration;
	}

	void iterate(){
		if(solver_type == "jacobi"){
			jacobi_fused_iteration(this->crs_mat, this->b, this->x_old, this->x_new);
			// jacobi_separate_iteration(this->crs_mat, this->D, this->b, this->x_old, this->x_new);
		}
		else if (solver_type == "gauss-seidel"){
			gs_fused_iteration(this->crs_mat, this->tmp, this->b, this->x);
			// gs_separate_iteration(this->crs_mat_U, this->crs_mat_L, this->tmp, this->D, this->b, this->x);
		}
	}

	void exchange_arrays(){
		if(solver_type == "jacobi"){
			std::swap(this->x_old, this->x_new);
		}
		else if (solver_type == "gauss-seidel"){
			// Nothing to exchange here
		}
	}

	void save_x_star(){
		if(solver_type == "jacobi"){
			std::swap(this->x_old, this->x_star);
		}
		else if (solver_type == "gauss-seidel"){
			std::swap(this->x, this->x_star);
		}
		compute_residual(this->crs_mat, this->x_star, this->b, this->residual, this->tmp);
		this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		this->collected_residual_norms[this->collected_residual_norms_count] = this->residual_norm;
	}

	void init_residual(){
		compute_residual(this->crs_mat, this->x_old, this->b, this->residual, this->tmp);
		this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
	}

	void init_stopping_criteria(){
		this->stopping_criteria = this->tolerance * this->residual_norm;
	}

	void record_residual_norm(){
		if(solver_type == "jacobi"){
			compute_residual(this->crs_mat, this->x_new, this->b, this->residual, this->tmp);
		}
		else if (solver_type == "gauss-seidel"){
			compute_residual(this->crs_mat, this->x, this->b, this->residual, this->tmp);
		}
		this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		this->collected_residual_norms[this->collected_residual_norms_count] = this->residual_norm;
	}

	void sample_residual(Stopwatch *per_iteration_time){
		if(this->iter_count % this->residual_check_len == 0){
			this->record_residual_norm();
			this->time_per_iteration[this->collected_residual_norms_count] = \
				per_iteration_time->check();
			++this->collected_residual_norms_count;
		}
	}

	~Solver(){
		delete crs_mat;
		delete[] x_star;
		delete[] x_new;
		delete[] x_old;
		delete[] x;
		delete[] b;
		delete[] D;
		delete[] tmp;
		delete[] collected_residual_norms;
	}

};

#endif