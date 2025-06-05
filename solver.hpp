#pragma once

#include "common.hpp"
#include "kernels.hpp"
#include "methods/bicgstab.hpp"
#include "methods/cg.hpp"
#include "methods/gauss_seidel.hpp"
#include "methods/gmres.hpp"
#include "methods/jacobi.hpp"
#include "methods/richardson.hpp"
#include "sparse_matrix.hpp"
#include "utilities/smax_helpers.hpp"

#include <float.h>

// clang-format off

class Solver
{
private:
	Args *cli_args;

public:
	MatrixCRS *crs_mat;
	MatrixCRS *crs_mat_L;
	MatrixCRS *crs_mat_L_strict;
	MatrixCRS *crs_mat_U;
	MatrixCRS *crs_mat_U_strict;
	SolverType method;
	PrecondType preconditioner = PrecondType::None;
#ifdef USE_SMAX
    SMAX::Interface *smax;
#endif

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
	bool gmres_restarted = false;

	// Vectors
	double *x_star; // General
	double *b; // General
	double *tmp; // General
	double *residual; // General
	double *D; // General
	double *x_0; // General
	double *x_new; // Richardson + Jacobi + CG + BiCGSTAB
	double *x_old; // Richardson + Jacobi + CG + GMRES + BiCGSTAB
	double *x; // GS + GMRES
	double *p_old; // CG + BiCGSTAB
	double *p_new; // CG + BiCGSTAB
	double *z_old; // CG
	double *z_new; // CG
	double *residual_0;
	double *residual_old; // CG + BiCGSTAB
	double *residual_new; // CG + BiCGSTAB
	double *v; // BiCGSTAB
	double *h; // BiCGSTAB
	double *s; // BiCGSTAB
	double *s_tmp; // BiCGSTAB
	double *t; // BiCGSTAB
	double *t_tmp; // BiCGSTAB
	double rho_old; // BiCGSTAB
	double rho_new; // BiCGSTAB
	double *z; // BiCGSTAB
	double *V; // GMRES
	double *Vy; // GMRES
	double *y; // GMRES + BiCGSTAB
	double *H; // GMRES
	double *H_tmp; // GMRES
	double *J; // GMRES
	double *Q; // GMRES
	double *Q_tmp; // GMRES
	double *w; // GMRES
	double *R; // GMRES
	double *g; // GMRES
	double *g_tmp; // GMRES
	double alpha; // Richardson 
	double beta; // GMRES 

	// Misc
	double *collected_residual_norms;
	double *time_per_iteration;

	Solver(Args *_cli_args) : cli_args(_cli_args) {
		method = cli_args->method;
		preconditioner = cli_args->preconditioner;

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

	// NOTE: We only initialize the structs needed for the solver
	// and preconditioner selected

	void allocate_structs(){
		x_star =     new double [this->crs_mat->n_cols];
		x_0 =        new double [this->crs_mat->n_cols];
		b =          new double [this->crs_mat->n_cols];
		tmp =        new double [this->crs_mat->n_cols];
		residual =   new double [this->crs_mat->n_cols];
		residual_0 = new double [this->crs_mat->n_cols];
		D =          new double [this->crs_mat->n_cols];

		// Solver-specific structs
		if(method == SolverType::Richardson){
			x_new = new double [this->crs_mat->n_cols];
			x_old = new double [this->crs_mat->n_cols];
		}
		else if(method == SolverType::Jacobi){
			x_new = new double [this->crs_mat->n_cols];
			x_old = new double [this->crs_mat->n_cols];
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){
			x = new double [this->crs_mat->n_cols];
		}
		else if (method == SolverType::ConjugateGradient){
			x_new = new double [this->crs_mat->n_cols];
			x_old = new double [this->crs_mat->n_cols];
			p_new = new double [this->crs_mat->n_cols];
			p_old = new double [this->crs_mat->n_cols];
			residual_new = new double [this->crs_mat->n_cols];
			residual_old = new double [this->crs_mat->n_cols];
			z_new = new double [this->crs_mat->n_cols];
			z_old = new double [this->crs_mat->n_cols];
		}
		else if (method == SolverType::GMRES){
			x =      new double [this->crs_mat->n_cols];
			x_old =  new double [this->crs_mat->n_cols];
			V =      new double [this->crs_mat->n_cols * (this->gmres_restart_len + 1)];
			Vy =     new double [this->crs_mat->n_cols];
			y =      new double [this->gmres_restart_len];
			H =      new double [(this->gmres_restart_len + 1) * this->gmres_restart_len];
			H_tmp =  new double [(this->gmres_restart_len + 1) * this->gmres_restart_len];
			J =      new double [(this->gmres_restart_len + 1) * (this->gmres_restart_len + 1)];
			Q =      new double [(this->gmres_restart_len + 1) * (this->gmres_restart_len + 1)];
			Q_tmp =  new double [(this->gmres_restart_len + 1) * (this->gmres_restart_len + 1)];
			w =      new double [this->crs_mat->n_cols];  // NOTE: Currently not used
			R =      new double [this->gmres_restart_len * (this->gmres_restart_len + 1)];
			g =      new double [this->gmres_restart_len + 1];
			g_tmp =  new double [this->gmres_restart_len + 1];
		}
		else if (method == SolverType::BiCGSTAB){
			x_new = new double [this->crs_mat->n_cols];
			x_old = new double [this->crs_mat->n_cols];
			p_new = new double [this->crs_mat->n_cols];
			p_old = new double [this->crs_mat->n_cols];
			residual_new = new double [this->crs_mat->n_cols];
			residual_old = new double [this->crs_mat->n_cols];
			v     = new double [this->crs_mat->n_cols];
			h     = new double [this->crs_mat->n_cols];
			s     = new double [this->crs_mat->n_cols];
			s_tmp = new double [this->crs_mat->n_cols];
			t     = new double [this->crs_mat->n_cols];
			t_tmp = new double [this->crs_mat->n_cols];
			y     = new double [this->crs_mat->n_cols];
			z     = new double [this->crs_mat->n_cols];
		}
	}

	void init_structs(){
		#pragma omp parallel for
		for(int i = 0; i < this->crs_mat->n_cols; ++i){
			tmp[i] =        0.0;
			residual[i] =   0.0;
			residual_0[i] = 0.0;
		}

		if(!this->gmres_restarted){
			// We don't want to overwrite these when restarting GMRES
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x_star[i] =     0.0;
				x_0[i] =        INIT_X_VAL;
				b[i] =          B_VAL;
				D[i] = 0.0;
			}
		}

		// Solver-specific structs
		if(method == SolverType::Richardson){
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x_new[i] = 0.0;
				x_old[i] = x_0[i];
			}

			this->alpha = 1.0 / infty_mat_norm(this->crs_mat);
			IF_DEBUG_MODE(printf("||A||_\\infty = %f\n", infty_mat_norm(this->crs_mat)))
		}
		else if(method == SolverType::Jacobi){
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x_new[i] = 0.0;
				x_old[i] = x_0[i];
			}
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){	
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x[i] = x_0[i];
			}
		}
		else if (method == SolverType::ConjugateGradient){
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x_new[i] = 0.0;
				x_old[i] = x_0[i];
				p_new[i] = 0.0;
				p_old[i] = 0.0;
				residual_new[i] = 0.0;
				residual_old[i] = 0.0;
				z_new[i] = 0.0;
				z_old[i] = 0.0;
			}
		}
		else if (method == SolverType::GMRES){
			// NOTE: We only want to copy x <- x_0 on the first invocation of this routine.
			// All other invocations will be due to resets, in which case the approximate x vector
			// will be explicity computed.
			if(!this->gmres_restarted){
				#pragma omp parallel for
				for(int i = 0; i < this->crs_mat->n_cols; ++i){
					x[i] = x_0[i];
					x_old[i] = x_0[i];
				}
			}

			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols * (this->gmres_restart_len + 1); ++i){
				V[i] = 0.0;
			}

			init_vector(Vy, 0.0, this->crs_mat->n_cols);
			init_vector(w, 0.0, this->crs_mat->n_cols); // NOTE: Currently not used
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
		else if (method == SolverType::BiCGSTAB){
			#pragma omp parallel for
			for(int i = 0; i < this->crs_mat->n_cols; ++i){
				x_new[i] = 0.0;
				x_old[i] = x_0[i];
				p_new[i] = 0.0;
				p_old[i] = 0.0;
				residual_new[i] = 0.0;
				residual_old[i] = 0.0;
				v[i]     = 0.0;
				h[i]     = 0.0;
				s[i]     = 0.0;
				s_tmp[i] = 0.0;
				t[i]     = 0.0;
				t_tmp[i] = 0.0;
			}
		}
	}

	void init_residual(){
		if(method == SolverType::Richardson){
			compute_residual(this->crs_mat, this->x_old, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if(method == SolverType::Jacobi){
			compute_residual(this->crs_mat, this->x_old, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){
			compute_residual(this->crs_mat, this->x, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if(method == SolverType::ConjugateGradient){
			compute_residual(this->crs_mat, this->x_old, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));

			// Precondition the initial residual
			IF_DEBUG_MODE_FINE(SanityChecker::print_vector(this->residual, this->crs_mat->n_cols, "residual before preconditioning"));
			apply_preconditioner(this->preconditioner, this->crs_mat_L_strict, this->crs_mat_U_strict, this->D, this->z_old, this->residual, this->tmp SMAX_ARGS(0, smax, "init M^{-1} * residual"));
			IF_DEBUG_MODE_FINE(SanityChecker::print_vector(this->z_old, this->crs_mat->n_cols, "residual after preconditioning"));

			// Make copies of initial residual for solver
			copy_vector(this->p_old, this->z_old, this->crs_mat->n_cols);
			copy_vector(this->residual_old, this->residual, this->crs_mat->n_cols);
			this->residual_norm = infty_vec_norm(this->z_old, this->crs_mat->n_cols);
		}
		else if (method == SolverType::GMRES){
			IF_DEBUG_MODE(SanityChecker::print_vector(this->x, this->crs_mat->n_cols, "old_x1"));
			compute_residual(this->crs_mat, this->x, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));

			// Precondition the initial residual
			IF_DEBUG_MODE(SanityChecker::print_vector(this->residual, this->crs_mat->n_cols, "residual before preconditioning"));
			apply_preconditioner(this->preconditioner, this->crs_mat_L, this->crs_mat_U, this->D, this->residual, this->residual, this->tmp);
			IF_DEBUG_MODE(SanityChecker::print_vector(this->residual, this->crs_mat->n_cols, "residual after preconditioning"));

			IF_DEBUG_MODE(SanityChecker::print_vector(this->x, this->crs_mat->n_cols, "old_x2"));
			this->residual_norm = euclidean_vec_norm(this->residual, this->crs_mat->n_cols);
			this->beta = this->residual_norm; // NOTE: Beta should be according to euclidean norm (Saad)
			
			this->g[0] = this->beta;
			this->g_tmp[0] = this->beta;

			// V[0] <- r / beta
			// i.e. The first row of V (orthonormal search vectors) gets scaled initial residual
			scale(this->V, this->residual, 1.0/this->beta, this->crs_mat->n_cols);
			IF_DEBUG_MODE(SanityChecker::print_vector(this->residual, this->crs_mat->n_cols, "init_residual"));
			IF_DEBUG_MODE(printf("||init_residual||_2 = %f\n", this->residual_norm))
			IF_DEBUG_MODE(SanityChecker::print_vector(this->V, this->crs_mat->n_cols, "init_v"));
		}
		else if (method == SolverType::BiCGSTAB){
			compute_residual(this->crs_mat, this->x_old, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));

			// Make copies of initial residual for solver
			copy_vector(this->p_old, this->residual, this->crs_mat->n_cols);
			copy_vector(this->residual_old, this->residual, this->crs_mat->n_cols);
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
			this->rho_old = dot(this->residual_old, this->residual, this->crs_mat->n_cols);
		}
		copy_vector(this->residual_0, this->residual, this->crs_mat->n_cols);
		this->collected_residual_norms[this->collected_residual_norms_count] = this->residual_norm;
		
	}

	void init_stopping_criteria(){
		this->stopping_criteria = this->tolerance * this->residual_norm;
	}

#ifdef USE_SMAX
	void register_structs(){
		int N = this->crs_mat->n_rows;

		// Register kernel tag, platform, and metadata
		if(method == SolverType::Richardson){
			register_spmv(this->smax, "residual_spmv", this->crs_mat, this->x_old, N, this->tmp, N);
			register_spmv(this->smax, "update_residual", this->crs_mat, this->x_new, N, this->tmp, N);
		}
		else if(method == SolverType::Jacobi){
			register_spmv(this->smax, "residual_spmv", this->crs_mat, this->x_old, N, this->tmp, N);
			register_spmv(this->smax, "x_new <- A*x_old", this->crs_mat, this->x_old, N, this->x_new, N);
		}
		else if (method == SolverType::GaussSeidel){	
			register_spmv(this->smax, "residual_spmv", this->crs_mat, this->x, N, this->tmp, N);
			register_spmv(this->smax, "tmp <- U*x", this->crs_mat_U_strict, this->x, N, this->tmp, N);
			register_sptrsv(this->smax, "solve x <- (D+L)^{-1}(b-Ux)", this->crs_mat_L, this->x, N, this->tmp, N);
		}
		else if (method == SolverType::SymmetricGaussSeidel){	
			register_spmv(this->smax, "residual_spmv", this->crs_mat, this->x, N, this->tmp, N);
			register_spmv(this->smax, "tmp <- U*x", this->crs_mat_U_strict, this->x, N, this->tmp, N);
			register_sptrsv(this->smax, "solve x <- (D+L)^{-1}(b-Ux)", this->crs_mat_L, this->x, N, this->tmp, N);
			register_spmv(this->smax, "tmp <- L*x", this->crs_mat_L_strict, this->x, N, this->tmp, N);
			register_sptrsv(this->smax, "solve x <- (U+L)^{-1}(b-Ux)", this->crs_mat_U, this->x, N, this->tmp, N, true);
		}
		else if (method == SolverType::ConjugateGradient){
			register_spmv(this->smax, "residual_spmv", this->crs_mat, this->x_old, N, this->tmp, N);
			register_spmv(this->smax, "tmp <- A*p_old", this->crs_mat, this->p_old, N, this->tmp, N);
			if (preconditioner == PrecondType::GaussSeidel) {
				register_sptrsv(this->smax, "init M^{-1} * residual", this->crs_mat_L, this->z_old, N, this->residual, N);
				register_sptrsv(this->smax, "M^{-1} * residual", this->crs_mat_L, this->z_new, N, this->residual_new, N);
			} else if (preconditioner == PrecondType::BackwardsGaussSeidel) {
				register_sptrsv(this->smax, "init M^{-1} * residual", this->crs_mat_U, this->z_old, N, this->residual, N, true);
				register_sptrsv(this->smax, "M^{-1} * residual", this->crs_mat_U, this->z_new, N, this->residual_new, N, true);
			} else if (preconditioner == PrecondType::SymmetricGaussSeidel) {
				register_sptrsv(this->smax, "init M^{-1} * residual_lower", this->crs_mat_L, this->tmp, N, this->residual, N);
				register_sptrsv(this->smax, "init M^{-1} * residual_upper", this->crs_mat_U, this->z_old, N, this->tmp, N, true);
				register_sptrsv(this->smax, "M^{-1} * residual_lower", this->crs_mat_L, this->tmp, N, this->residual_new, N);
				register_sptrsv(this->smax, "M^{-1} * residual_upper", this->crs_mat_U, this->z_new, N, this->tmp, N, true);
			}
		}
		else if (method == SolverType::GMRES){
			register_spmv(this->smax, "residual_spmv", this->crs_mat, this->x, N, this->tmp, N);
			register_spmv(this->smax, "w_j <- A*v_j", this->crs_mat, this->V, N * (this->gmres_restart_len + 1), this->tmp, N);
		}
		else if (method == SolverType::BiCGSTAB){
			register_spmv(this->smax, "residual_spmv", this->crs_mat, this->x_old, N, this->tmp, N);
			register_spmv(this->smax, "v <- A*y", this->crs_mat, this->y, N, this->v, N);
			register_spmv(this->smax, "z <- A*s_tmp", this->crs_mat, this->s_tmp, N, this->z, N);
		}
	}
#endif

	void iterate(
		Timers *timers
	){
		if(method == SolverType::Richardson){
			richardson_separate_iteration(timers, this->crs_mat, this->b, this->x_new, this->x_old, this->tmp, this->residual, this->alpha
				SMAX_ARGS(this->smax)
			);
		}
		else if(method == SolverType::Jacobi){
			// jacobi_fused_iteration(this->crs_mat, this->b, this->x_new, this->x_old);
			jacobi_separate_iteration(
				timers,
				this->crs_mat,
				this->D,
				this->b,
				this->x_new,
				this->x_old
				SMAX_ARGS(this->smax));
		}
		else if (method == SolverType::GaussSeidel){
			// gs_fused_iteration(this->crs_mat, this->b, this->x);
			gs_separate_iteration(
				timers,
				this->crs_mat_U_strict,
				this->crs_mat_L_strict,
				this->tmp,
				this->D,
				this->b,
				this->x
				SMAX_ARGS(this->smax));
		}
		else if (method == SolverType::SymmetricGaussSeidel){
			// gs_fused_iteration(this->crs_mat, this->b, this->x);
			gs_separate_iteration(
				timers, 
				this->crs_mat_U_strict, 
				this->crs_mat_L_strict, 
				this->tmp, 
				this->D, 
				this->b, 
				this->x
				SMAX_ARGS(this->smax));
			bgs_separate_iteration(
				timers, 
				this->crs_mat_U_strict, 
				this->crs_mat_L_strict, 
				this->tmp, 
				this->D, 
				this->b, 
				this->x
				SMAX_ARGS(this->smax));
		}
		else if(method == SolverType::ConjugateGradient){
			cg_separate_iteration(
				timers,
				this->preconditioner,
				this->crs_mat,
				this->crs_mat_L_strict,
				this->crs_mat_U_strict,
				this->D,
				this->x_new,
				this->x_old,
				this->tmp,
				this->p_new,
				this->p_old,
				this->residual_new,
				this->residual_old,
				this->z_new,
				this->z_old
				SMAX_ARGS(this->smax)
			);
			std::swap(this->residual, this->residual_new);
		}
		else if (method == SolverType::GMRES){
			gmres_separate_iteration(
				timers,
				this->preconditioner,
				this->crs_mat,
				this->crs_mat_L_strict,
				this->crs_mat_U_strict,
				this->D,
				this->iter_count,
				this->gmres_restart_count,
				this->gmres_restart_len,
				this->residual_norm,
				this->V,
				this->H,
				this->H_tmp,
				this->J,
				this->Q,
				this->Q_tmp,
				this->tmp, // TODO: Should this be a fresh w?
				this->R,
				this->g,
				this->g_tmp,
				this->b,
				this->x,
				this->tmp,
				this->beta
				SMAX_ARGS(this->smax)
			);
		}
		else if (method == SolverType::BiCGSTAB){
			bicgstab_separate_iteration(
				timers,
				this->preconditioner,
				this->crs_mat,
				this->crs_mat_L_strict,
				this->crs_mat_U_strict,
				this->D,
				this->x_new,
				this->x_old,
				this->tmp,
				this->p_new,
				this->p_old,
				this->residual_new,
				this->residual_old,
				this->residual_0,
				this->v,
				this->h,
				this->s,
				this->s_tmp,
				this->t,
				this->t_tmp,
				this->y,
				this->z,
				this->rho_new,
				this->rho_old
				SMAX_ARGS(this->smax)
			);
			std::swap(this->residual, this->residual_new);
		}
	}

void exchange(){
		if(method == SolverType::Richardson){
			std::swap(this->x_old, this->x_new);
		}
		else if(method == SolverType::Jacobi){
			std::swap(this->x_old, this->x_new);
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){
			// Nothing to exchange
		}
		else if(method == SolverType::ConjugateGradient){
			std::swap(this->p_old, this->p_new);
			std::swap(this->z_old, this->z_new);
			std::swap(this->residual_old, this->residual); // <- swapped r and r_new earlier
			std::swap(this->x_old, this->x_new);
		}
		else if (method == SolverType::GMRES){
			// Nothing to exchange
		}
		else if (method == SolverType::BiCGSTAB){
			std::swap(this->p_old, this->p_new);
			std::swap(this->residual_old, this->residual); // <- swapped r and r_new earlier
			std::swap(this->x_old, this->x_new);
			std::swap(this->rho_old, this->rho_new);
		}
#ifdef USE_SMAX
		// In order to maintain consistent view of pointers inside of library
		if(method == SolverType::Richardson){
			auto *residual_spmv = dynamic_cast<SMAX::KERNELS::SpMVKernel *>(smax->kernel("residual_spmv"));
			auto *update_spmv = dynamic_cast<SMAX::KERNELS::SpMVKernel *>(smax->kernel("update_residual"));
			std::swap(residual_spmv->args->x, update_spmv->args->x);
		}
		else if(method == SolverType::Jacobi){
			this->smax->kernel("x_new <- A*x_old")->swap_operands();
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){
			// Nothing to swap or copy
		}
		else if(method == SolverType::ConjugateGradient){
			auto *spmv = dynamic_cast<SMAX::KERNELS::SpMVKernel *>(smax->kernel("tmp <- A*p_old"));
			spmv->args->x->val = static_cast<void *>(this->p_old);
			if (preconditioner == PrecondType::GaussSeidel || preconditioner == PrecondType::BackwardsGaussSeidel) {
				auto *sptrsv = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * residual"));
				sptrsv->args->x->val = static_cast<void *>(this->z_new);
				sptrsv->args->y->val = static_cast<void *>(this->residual_new);
			}
			else if(preconditioner == PrecondType::SymmetricGaussSeidel){
				auto *lower_sptrsv = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * residual_lower"));
				lower_sptrsv->args->x->val = static_cast<void *>(this->tmp);
				lower_sptrsv->args->y->val = static_cast<void *>(this->residual_new);\

				auto *upper_sptrsv = dynamic_cast<SMAX::KERNELS::SpTRSVKernel *>(smax->kernel("M^{-1} * residual_upper"));
				upper_sptrsv->args->x->val = static_cast<void *>(this->z_new);
				upper_sptrsv->args->y->val = static_cast<void *>(this->tmp);
			}
		}
		else if (method == SolverType::GMRES){
			// Nothing to swap or copy
		}
		else if (method == SolverType::BiCGSTAB){
			// Nothing to swap or copy
		}
#endif
	}

	// NOTE: Anything to update for SMAX?
	void save_x_star(){
		IF_DEBUG_MODE(printf("Saving x*\n"))
		if(method == SolverType::Richardson){
			std::swap(this->x_old, this->x_star);
		}
		else if(method == SolverType::Jacobi){
			std::swap(this->x_old, this->x_star);
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){
			std::swap(this->x, this->x_star);
		}
		else if (method == SolverType::ConjugateGradient){
			std::swap(this->x_old, this->x_star);
		}
		else if (method == SolverType::GMRES){
			this->get_explicit_x();
			std::swap(this->x, this->x_star);
		}
		else if (method == SolverType::BiCGSTAB){
			std::swap(this->x_old, this->x_star);
		}

		compute_residual(this->crs_mat, this->x_star, this->b, this->residual, this->tmp
			SMAX_ARGS(this->smax, "residual_spmv"));
		this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		this->collected_residual_norms[this->collected_residual_norms_count] = this->residual_norm;
	}

	void record_residual_norm(){
		if(method == SolverType::Richardson){
			// Residual vector is updated implicitly, so do not need to compute it
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if(method == SolverType::Jacobi){
			compute_residual(this->crs_mat, this->x_new, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){
			compute_residual(this->crs_mat, this->x, this->b, this->residual, this->tmp
				SMAX_ARGS(this->smax, "residual_spmv"));
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (method == SolverType::ConjugateGradient){
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		else if (method == SolverType::GMRES){
			// Nothing to do here, since residual vector norm is computed implicitly
		}
		else if (method == SolverType::BiCGSTAB){
			this->residual_norm = infty_vec_norm(this->residual, this->crs_mat->n_cols);
		}
		
		this->collected_residual_norms[this->collected_residual_norms_count + 1] = this->residual_norm;
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
		if(method == SolverType::GMRES){

			double diag_elem = 1.0;
	
			// Adjust for restarting
			int n_solver_iters = this->iter_count;
			n_solver_iters -= this->gmres_restart_count * this->gmres_restart_len;
			IF_DEBUG_MODE(SanityChecker::print_gmres_iter_counts(n_solver_iters, this->gmres_restart_count))

			// Backward triangular solve y <- R^{-1}(g) [(m+1 x m)(m x 1) = (m+1 x 1)]
			// Traverse R \in \mathbb{R}^(m+1 x m) from last to first row
			for(int row_idx = n_solver_iters - 1; row_idx >= 0; --row_idx){
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
		if(method == SolverType::GMRES){
			bool norm_convergence = this->residual_norm < this->stopping_criteria;
			bool over_max_iters = this->iter_count > this->max_iters;
			bool restart_cycle_reached = (this->iter_count) % (this->gmres_restart_len) == 0;
			if(!norm_convergence && !over_max_iters && restart_cycle_reached){
				this->gmres_restarted = true;

				IF_DEBUG_MODE(printf("GMRES restart: %i\n", this->gmres_restart_count))
				// x <- x_0 + Vy
				this->get_explicit_x();
				copy_vector(this->x_old, this->x, this->crs_mat->n_cols);

				// Re-initialize relevant data structures after restarting GMRES
				// NOTE: x is the only struct which is not re-initialized
				this->init_structs();

				// TODO: This shouldn't be necessary
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
		delete crs_mat_L_strict;
		delete crs_mat_U_strict;
		delete[] collected_residual_norms;
		delete[] time_per_iteration;
		delete[] x_star;
		delete[] x_0;
		delete[] b;
		delete[] D;
		delete[] tmp;
		delete[] residual;
		delete[] residual_0;

		// Solver-specific structs
		if(method == SolverType::Richardson){
			delete[] x_new;
			delete[] x_old;
		}
		else if(method == SolverType::Jacobi){
			delete[] x_new;
			delete[] x_old;
		}
		else if (method == SolverType::GaussSeidel || method == SolverType::SymmetricGaussSeidel){
			delete[] x;
		}
		else if(method == SolverType::ConjugateGradient){
			delete[] x_new;
			delete[] x_old;
			delete[] p_new;
			delete[] p_old;
			delete[] residual_new;
			delete[] residual_old;
			delete[] z_new;
			delete[] z_old;
		}
		else if (method == SolverType::GMRES){
			delete[] x;
			delete[] x_old;
			delete[] V;
			delete[] Vy;
			delete[] y;
			delete[] H;
			delete[] H_tmp;
			delete[] J;
			delete[] Q;
			delete[] Q_tmp;
			delete[] R;
			delete[] w;
			delete[] g;
			delete[] g_tmp;
		}
		else if (method == SolverType::BiCGSTAB){
			delete[] x_new;
			delete[] x_old;
			delete[] p_new;
			delete[] p_old;
			delete[] residual_new;
			delete[] residual_old;
			delete[] v;
			delete[] h;
			delete[] s;
			delete[] s_tmp;
			delete[] t;
			delete[] t_tmp;
			delete[] y;
			delete[] z;
		}
	}

};

// clang-format on
