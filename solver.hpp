#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "common.hpp"
#include "sparse_matrix.hpp"
#include "kernels.hpp"
#include "methods/jacobi.hpp"
#include "methods/gauss_seidel.hpp"
#include "methods/cg.hpp"
#include "methods/gmres.hpp"
#include "methods/bicgstab.hpp"

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
	double stopping_criteria = 0.0;
	int iter_count = 0;
	int gmres_restart_count = 0;
	int collected_residual_norms_count = 0;
	double residual_norm = DBL_MAX;
	// Assign config.mk parameters to the solver parameters
	int max_iters = MAX_ITERS;
	double tolerance = TOL;
	int residual_check_len = RES_CHECK_LEN;
	int gmres_restart_len = GMRES_RESTART_LEN;

	// Flags
	bool convergence_flag = false;
	bool gmres_copy_x_0 = true;

	// Vectors
	double *x_star; // General
	double *b; // General
	double *tmp; // General
	double *residual; // General
	double *D; // General
	double *x_0; // General
	double *x_new; // Jacobi + CG
	double *x_old; // Jacobi + CG + GMRES
	double *x; // GS + GMRES
	double *p_old; // CG
	double *p_new; // CG
	double *residual_old; // CG
	double *residual_new; // CG
	double *V; // GMRES
	double *Vy; // GMRES
	double *y; // GMRES
	double *H; // GMRES
	double *H_tmp; // GMRES
	double *J; // GMRES
	double *Q; // GMRES
	double *Q_tmp; // GMRES
	double *w; // GMRES
	double *R; // GMRES
	double *g; // GMRES
	double *g_tmp; // GMRES
	double gmres_beta; // GMRES 

	// Misc
	double *collected_residual_norms;
	double *time_per_iteration;

	Solver(Args *_cli_args) : cli_args(_cli_args) {
		solver_type = cli_args->solver_type;
		preconditioner_type = cli_args->preconditioner_type;

		collected_residual_norms = new double[this->max_iters];
		time_per_iteration = new double[this->max_iters];

		for(int i = 0; i < this->max_iters; ++i){
			collected_residual_norms[i] = 0.0;
			time_per_iteration[i] = 0.0;
		}
	}

	bool check_stopping_criteria(){
		bool norm_convergence = this->residual_norm < this->stopping_criteria;
		bool over_max_iters = this->iter_count >= this->max_iters;
		bool divergence = this->residual_norm > DBL_MAX;
		IF_DEBUG_MODE_FINE(if(norm_convergence) printf("norm convergence met: %f < %f\n", this->residual_norm, this->stopping_criteria))
		IF_DEBUG_MODE_FINE(if(over_max_iters) printf("over max iters: %i >= %i\n", this->iter_count, this->max_iters))
		IF_DEBUG_MODE_FINE(if(divergence) printf("divergence\n"))
		return norm_convergence || over_max_iters || divergence;
	}

	void apply_preconditioner(){
		// TODO
	}

	// NOTE: We only initialize the structs needed for the solver
	// and preconditioner selected
	void allocate_structs(){
		x_star =   new double [this->crs_mat->n_cols];
		x_0 =   new double [this->crs_mat->n_cols];
		b =        new double [this->crs_mat->n_cols];
		tmp =      new double [this->crs_mat->n_cols];
		residual = new double [this->crs_mat->n_cols];
		D =        new double [this->crs_mat->n_cols];

		// Solver-specific structs
		if(solver_type == "jacobi"){
			x_new = new double [this->crs_mat->n_cols];
			x_old = new double [this->crs_mat->n_cols];
		}
		else if (solver_type == "gauss-seidel"){
			x = new double [this->crs_mat->n_cols];
		}
		else if (solver_type == "conjugate-gradient"){
			x_new = new double [this->crs_mat->n_cols];
			x_old = new double [this->crs_mat->n_cols];
			p_new = new double [this->crs_mat->n_cols];
			p_old = new double [this->crs_mat->n_cols];
			residual_new = new double [this->crs_mat->n_cols];
			residual_old = new double [this->crs_mat->n_cols];
		}
		else if (solver_type == "gmres"){
			x = 		 new double [this->crs_mat->n_cols];
			x_old =  new double [this->crs_mat->n_cols];
			V =      new double [this->crs_mat->n_cols * (this->gmres_restart_len + 1)];
			Vy =     new double [this->crs_mat->n_cols];
			y =      new double [this->gmres_restart_len];
			H =      new double [(this->gmres_restart_len + 1) * this->gmres_restart_len];
			H_tmp =  new double [(this->gmres_restart_len + 1) * this->gmres_restart_len];
			J =      new double [(this->gmres_restart_len + 1) * (this->gmres_restart_len + 1)];
			Q =      new double [(this->gmres_restart_len + 1) * (this->gmres_restart_len + 1)];
			Q_tmp =  new double [(this->gmres_restart_len + 1) * (this->gmres_restart_len + 1)];
			R =      new double [this->gmres_restart_len * (this->gmres_restart_len + 1)];
			g =      new double [this->gmres_restart_len + 1];
			g_tmp =  new double [this->gmres_restart_len + 1];
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}
	}

	void init_structs(){
		#pragma omp parallel for
		for(int i = 0; i < this->crs_mat->n_cols; ++i){
			x_star[i] =   0.0;
			x_0[i] =      INIT_X_VAL;
			b[i] =        B_VAL;
			tmp[i] =      0.0;
			residual[i] = 0.0;
			D[i] =        0.0;
		}

		// Solver-specific structs
		if(solver_type == "jacobi"){
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x_new[i] = 0.0;
				x_old[i] = x_0[i];
			}
		}
		else if (solver_type == "gauss-seidel"){	
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x[i] = x_0[i];
			}
		}
		else if (solver_type == "conjugate-gradient"){
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x_new[i] = 0.0;
				x_old[i] = x_0[i];
				p_new[i] = 0.0;
				p_old[i] = 0.0;
				residual_new[i] = 0.0;
				residual_old[i] = 0.0;
			}
		}
		else if (solver_type == "gmres"){
			// NOTE: We only want to copy x <- x_0 on the first invocation of this routine.
			// All other invocations will be due to resets, in which case the approximate x vector
			// will be explicity computed.
			if(this->gmres_copy_x_0){
				#pragma omp parallel for
				for(int i = 0; i < this->crs_mat->n_cols; ++i){
					x[i] = x_0[i];
					x_old[i] = x_0[i];
				}
				this->gmres_copy_x_0 = false;
			}

			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols * (this->gmres_restart_len + 1); ++i){
				V[i] = 0.0;
			}

			init_vector(Vy, 0.0, this->crs_mat->n_cols);
			init_vector(y, 0.0, this->gmres_restart_len);
			init_vector(g, 0.0, (this->gmres_restart_len + 1));
			init_vector(g_tmp, 0.0, (this->gmres_restart_len + 1));
			// init_dense_identity_matrix(H, (this->gmres_restart_len + 1), this->gmres_restart_len);
			init_vector(H, 0.0, (this->gmres_restart_len + 1) * this->gmres_restart_len);

			// init_dense_identity_matrix(H_tmp, (this->gmres_restart_len + 1), this->gmres_restart_len);
			init_vector(H_tmp, 0.0, (this->gmres_restart_len + 1) * this->gmres_restart_len);

			init_dense_identity_matrix(J, (this->gmres_restart_len + 1), (this->gmres_restart_len + 1));
			init_vector(R, 0.0, (this->gmres_restart_len + 1) * this->gmres_restart_len);

			// init_dense_identity_matrix(R, this->gmres_restart_len, (this->gmres_restart_len + 1));
			init_dense_identity_matrix(Q, (this->gmres_restart_len + 1), (this->gmres_restart_len + 1));
			init_dense_identity_matrix(Q_tmp, (this->gmres_restart_len + 1), (this->gmres_restart_len + 1));
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}
	}

	void init_residual(){
		if(solver_type == "jacobi"){
			compute_residual(this->crs_mat, this->x_old, this->b, this->residual, this->tmp);
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (solver_type == "gauss-seidel"){
			compute_residual(this->crs_mat, this->x, this->b, this->residual, this->tmp);
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if(solver_type == "conjugate-gradient"){
			compute_residual(this->crs_mat, this->x_old, this->b, this->residual, this->tmp);
			// Make copies of initial residual for solver
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				this->p_old[i] = this->residual[i];
				this->residual_old[i] = this->residual[i];
			}
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (solver_type == "gmres"){
			IF_DEBUG_MODE(SanityChecker::print_vector(this->x, this->crs_mat->n_cols, "old_x1"));

			compute_residual(this->crs_mat, this->x, this->b, this->residual, this->tmp);
			IF_DEBUG_MODE(SanityChecker::print_vector(this->x, this->crs_mat->n_cols, "old_x2"));
			this->residual_norm = euclidean_vec_norm(this->residual, this->crs_mat->n_cols);
			this->gmres_beta = this->residual_norm; // NOTE: Beta should be according to euclidean norm (Saad)
			
			this->g[0] = this->gmres_beta;
			this->g_tmp[0] = this->gmres_beta;

			// V[0] <- r / beta
			// i.e. The first row of V (orthonormal search vectors) gets scaled initial residual
			scale(this->V, this->residual, 1.0/this->gmres_beta, this->crs_mat->n_cols);
			IF_DEBUG_MODE(SanityChecker::print_vector(this->residual, this->crs_mat->n_cols, "init_residual"));
			IF_DEBUG_MODE(printf("||init_residual||_2 = %f\n", this->residual_norm))
			IF_DEBUG_MODE(SanityChecker::print_vector(this->V, this->crs_mat->n_cols, "init_v"));
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}
	}

	void init_stopping_criteria(){
		this->stopping_criteria = this->tolerance * this->residual_norm;
	}

	void iterate(
		Timers *timers
	){
		if(solver_type == "jacobi"){
			// jacobi_fused_iteration(this->crs_mat, this->b, this->x_new, this->x_old);
			jacobi_separate_iteration(timers, this->crs_mat, this->D, this->b,  this->x_new, this->x_old);
		}
		else if (solver_type == "gauss-seidel"){
			// gs_fused_iteration(this->crs_mat, this->b, this->x);
			gs_separate_iteration(timers, this->crs_mat_U, this->crs_mat_L, this->tmp, this->D, this->b, this->x);
		}
		else if(solver_type == "conjugate-gradient"){
			cg_separate_iteration(
				timers,
				this->crs_mat,
				this->x_new,
				this->x_old,
				this->tmp,
				this->p_new,
				this->p_old,
				this->residual_new,
				this->residual_old
			);
			std::swap(this->residual, this->residual_new);
		}
		else if (solver_type == "gmres"){
			gmres_separate_iteration(
				timers,
				this->crs_mat,
				this->iter_count,
				this->gmres_restart_count,
				this->gmres_restart_len,
				this->residual_norm,
				this->D,
				this->V,
				this->H,
				this->H_tmp,
				this->J,
				this->Q,
				this->Q_tmp,
				this->tmp,
				this->R,
				this->g,
				this->g_tmp,
				this->b,
				this->x,
				this->gmres_beta
			);
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}
	}

	void exchange_arrays(){
		if(solver_type == "jacobi"){
			std::swap(this->x_old, this->x_new);
		}
		else if (solver_type == "gauss-seidel"){
			// Nothing to exchange
		}
		else if(solver_type == "conjugate-gradient"){
			std::swap(this->p_old, this->p_new);
			std::swap(this->residual_old, this->residual); // <- swapped r and r_new earlier
			std::swap(this->x_old, this->x_new);
		}
		else if (solver_type == "gmres"){
			// Nothing to exchange
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}
	}

	void save_x_star(){
		IF_DEBUG_MODE(printf("Saving x*\n"))
		if(solver_type == "jacobi"){
			std::swap(this->x_old, this->x_star);
			compute_residual(this->crs_mat, this->x_star, this->b, this->residual, this->tmp);
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (solver_type == "gauss-seidel"){
			std::swap(this->x, this->x_star);
			compute_residual(this->crs_mat, this->x_star, this->b, this->residual, this->tmp);
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (solver_type == "gmres"){
			this->get_explicit_x();
			std::swap(this->x, this->x_star);
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}

		this->collected_residual_norms[this->collected_residual_norms_count] = this->residual_norm;
	}

	void record_residual_norm(){
		if(solver_type == "jacobi"){
			compute_residual(this->crs_mat, this->x_new, this->b, this->residual, this->tmp);
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (solver_type == "gauss-seidel"){
			compute_residual(this->crs_mat, this->x, this->b, this->residual, this->tmp);
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (solver_type == "gmres"){
			// TODO
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}
		
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

	void get_explicit_x(){
		// NOTE: Only relevant for GMRES, so we don't worry about other solvers
		if(solver_type == "gmres"){

			double diag_elem = 1.0;
	
			// Adjust for restarting
			int n_solver_iters = this->iter_count;
			n_solver_iters -= this->gmres_restart_count * this->gmres_restart_len;
			IF_DEBUG_MODE(SanityChecker::print_gmres_iter_counts(n_solver_iters, this->gmres_restart_count))

			// Backward triangular solve y <- R^{-1}(g) [(m+1 x m)(m x 1) = (m+1 x 1)]
			// Traverse R \in \mathbb{R}^(m+1 x m) from last to first row
			for(int row_idx = n_solver_iters; row_idx >= 0; --row_idx){
				double sum = 0.0;
				for(int col_idx = row_idx; col_idx < this->gmres_restart_len; ++col_idx){
						if(row_idx == col_idx){
								diag_elem = this->R[(row_idx*this->gmres_restart_len) + col_idx];
						}
						else{
								sum += this->R[(row_idx*this->gmres_restart_len) + col_idx] * this->y[col_idx];
						}
						
				}
				this->y[row_idx] = (this->g[row_idx] - sum) / diag_elem;
#ifdef DEBUG_MODE_FINE
        std::cout << g[row_idx] << " - " << sum << " / " << diag_elem << std::endl; 
#endif
			}

			// TODO: Change to appropriate dgemv routine
			// Vy <- V*y [(m x 1) = (m x n)(n x 1)]
			// dgemm_transpose1(this->V, this->y, this->Vy, (this->gmres_restart_len + 1), this->crs_mat->n_cols, 1);
			dgemm_transpose1(this->V, this->y, this->Vy, this->crs_mat->n_cols, this->gmres_restart_len, 1);

			// dense_MMM_t<VT>(V, &y[0], Vy, n_rows, restart_len, 1);


			IF_DEBUG_MODE_FINE(SanityChecker::print_vector(this->Vy, this->crs_mat->n_cols, "Vy"));


			// Finally, compute x <- x_0 + Vy [(n x 1) = (n x 1) + (n x m)(m x 1)]
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				this->x[i] = this->x_old[i] + this->Vy[i];
#ifdef DEBUG_MODE_FINE
        std::cout << "x[" << i << "] = " << x_old[i] << " + " << Vy[i] << " = " << x[i] << std::endl; 
#endif
			}

			IF_DEBUG_MODE_FINE(SanityChecker::print_vector(this->x, this->crs_mat->n_cols, "new_x"));
		}
	}

	void check_restart(){
		// NOTE: Only relevant for GMRES, so we don't worry about other solvers
		if(solver_type == "gmres"){
			bool norm_convergence = this->residual_norm < this->stopping_criteria;
			bool over_max_iters = this->iter_count >= this->max_iters;
			bool restart_cycle_reached = (this->iter_count + 1) % (this->gmres_restart_len) == 0;
			if(!norm_convergence && !over_max_iters && restart_cycle_reached){

				IF_DEBUG_MODE(printf("GMRES restart: %i\n", this->gmres_restart_count))
				// x <- x_0 + Vy
				this->get_explicit_x();
				copy_vector(this->x_old, this->x, this->crs_mat->n_cols);

				// Re-initialize relevant data structures after restarting GMRES
				// NOTE: x is the only struct which is not re-initialized
				this->init_structs();

				// Re-initialize residual with new inital x approximation
				this->init_residual();

				++this->gmres_restart_count;
			}
		}
	}



	~Solver(){
		// General structs
		delete crs_mat;
		delete crs_mat_L;
		delete crs_mat_U;
		delete[] collected_residual_norms;
		delete[] time_per_iteration;
		delete[] x_star;
		delete[] x_0;
		delete[] b;
		delete[] D;
		delete[] tmp;
		delete[] residual;

		// Solver-specific structs
		if(solver_type == "jacobi"){
			delete[] x_new;
			delete[] x_old;
		}
		else if (solver_type == "gauss-seidel"){
			delete[] x;
		}
		else if(solver_type == "conjugate-gradient"){
			delete[] x_new;
			delete[] x_old;
			delete[] p_new;
			delete[] p_old;
			delete[] residual_new;
			delete[] residual_old;
		}
		else if (solver_type == "gmres"){
			delete[] V;
			delete[] Vy;
			delete[] y;
			delete[] H;
			delete[] H_tmp;
			delete[] J;
			delete[] Q;
			delete[] Q_tmp;
			delete[] R;
			delete[] g;
			delete[] g_tmp;
			delete[] x;
			delete[] x_old;
		}
		else if (solver_type == "bicgstab"){
			// TODO
		}
	}

};

#endif