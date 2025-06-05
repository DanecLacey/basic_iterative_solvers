# basic_iterative_solvers #

<p align="center">
  <img src="figs/HPCG_compare_convergence.png" width="45%">
  <img src="figs/HPCG_compare_time_per_iter.png" width="45%">
</p>


This is a set of (preconditioned) iterative solvers for `Ax = b`, where `A` is typically assumed to be a large sparse matrix.

### Usage Examples ###
```bash
./basic_iterative_solvers HPCG-128.mtx -cg
```
```bash
./basic_iterative_solvers Anderson,Lx=100,Ly=100,Lz=50,ranpot=5.0 -gm -p gs
```

### Building basic_iterative_solvers ###
``` bash
git clone git@github.com:DanecLacey/basic_iterative_solvers.git
cd basic_iterative_solvers
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=<CXX> -DCMAKE_PREFIX_PATH=install_path (if using TPLs)
make
```

### Features ###
* Stacked timers around key code regions
* Optional third party libraries:
  * Instrumentation with [likwid](https://github.com/RRZE-HPC/likwid) markers for collecting hardware performance counters already built-in around native kernels
  * Matrices can be read from `.mtx` files, or generated with the [SCAMAC library](https://alvbit.bitbucket.io/scamac_docs/index.html) 
  * Integration with [SmaxKernels library](https://github.com/DanecLacey/SmaxKernels) for optimized sparse kernel implementations

### Supported Solvers ###
* **Jacobi** `-j`
* (Forward/Symmetric)**Gauss-Seidel** `-gs / -sgs`
* (Preconditioned) **Conjugate-Gradient** `-cg`
* (Preconditioned) (Restarted) **GMRES** `-gm`
* (Preconditioned) **BiCGSTAB** `-bi`

### Supported Preconditioners ###
* **Jacobi** `-p j`
* (Forward/Backward/Symmetric) **Gauss-Seidel** `-p (gs/bgs/sgs)`

### Notes ###
* The sparse matrix storage format of `A` is **CRS**.
* Only **left-preconditioning** is implemented.
* This code is mainly for **performance investigations**.