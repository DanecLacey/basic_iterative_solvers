#pragma once

#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "utilities/mmio.hpp"

#ifdef USE_SCAMAC
#include "scamac.h"
#endif

#ifdef USE_FAST_MMIO
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
namespace fmm = fast_matrix_market;
#endif

inline void sort_perm(int *arr, int *perm, int len, bool rev = false) {
    if (rev == false) {
        std::stable_sort(perm + 0, perm + len, [&](const int &a, const int &b) {
            return (arr[a] < arr[b]);
        });
    } else {
        std::stable_sort(perm + 0, perm + len, [&](const int &a, const int &b) {
            return (arr[a] > arr[b]);
        });
    }
}

template <typename IT>
std::vector<IT> compute_sort_permutation(const std::vector<IT> &rows,
                                         const std::vector<IT> &cols) {
    std::vector<IT> perm(rows.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](std::size_t i, std::size_t j) {
                         if (rows[i] != rows[j])
                             return rows[i] < rows[j];
                         if (cols[i] != cols[j])
                             return cols[i] < cols[j];
                         return false;
                     });
    return perm;
}

template <typename IT, typename VT>
std::vector<VT> apply_permutation(std::vector<IT> &perm,
                                  std::vector<VT> &original) {
    std::vector<VT> sorted;
    sorted.reserve(original.size());
    std::transform(perm.begin(), perm.end(), std::back_inserter(sorted),
                   [&](auto i) { return original[i]; });
    original = std::vector<VT>();
    return sorted;
}

struct MatrixCRS {
    int n_rows{};
    int n_cols{};
    int nnz{};

    int *row_ptr = nullptr;
    int *col = nullptr;
    double *val = nullptr;
#ifdef USE_SMAX
    int *perm = nullptr;
    int *inv_perm = nullptr;
#endif

    // Default Constructor
    MatrixCRS() = default;

    // Parameterized Constructor
    MatrixCRS(std::size_t num_rows, std::size_t num_cols, std::size_t num_nnz)
        : n_rows(num_rows), n_cols(num_cols), nnz(num_nnz) {
        // Allocate memory for vectors based on dimensions
        // row_ptr needs n_rows + 1 elements
        row_ptr = new int[n_rows + 1];
        col = new int[nnz];
        val = new double[nnz];

#ifdef USE_SMAX
        // Permutation arrays are typically sized by n_rows
        perm = new int[n_rows];
        inv_perm = new int[n_rows];
#endif
    }

    void print(void) {
        std::cout << "n_rows = " << n_rows << std::endl;
        std::cout << "n_cols = " << n_cols << std::endl;
        std::cout << "nnz = " << nnz << std::endl;

        std::cout << "row_ptr = [";
        for (int i = 0; i < n_rows + 1; ++i) {
            std::cout << row_ptr[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "col = [";
        for (int i = 0; i < nnz; ++i) {
            std::cout << col[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "values = [";
        for (int i = 0; i < nnz; ++i) {
            std::cout << static_cast<double>(val[i]) << ", ";
        }
        std::cout << "]" << std::endl;

#ifdef USE_SMAX
        std::cout << "perm = [";
        for (int i = 0; i < n_rows; ++i) {
            std::cout << perm[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "inv_perm = [";
        for (int i = 0; i < n_rows; ++i) {
            std::cout << inv_perm[i] << ", ";
        }
        std::cout << "]" << std::endl;
#endif
    }

    ~MatrixCRS() {
        delete[] row_ptr;
        delete[] col;
        delete[] val;
#ifdef USE_SMAX
        delete[] perm;
        delete[] inv_perm;
#endif
    }
};

struct MatrixCOO {
    long n_rows{};
    long n_cols{};
    long nnz{};

    bool is_sorted{};
    bool is_symmetric{};

    std::vector<int> I;
    std::vector<int> J;
    std::vector<double> values;

    // Default constructor (important for std::make_unique)
    MatrixCOO() = default;

    // Constructor to preallocate (optional, but good practice for known sizes)
    MatrixCOO(long rows, long cols, long num_non_zeros)
        : n_rows(rows), n_cols(cols), nnz(num_non_zeros) {
        I.reserve(nnz);
        J.reserve(nnz);
        values.reserve(nnz);
    }

    void write_to_mtx(std::string file_out_name) {
        std::string file_name = file_out_name + ".mtx";

        for (int nz_idx = 0; nz_idx < nnz; ++nz_idx) {
            ++I[nz_idx];
            ++J[nz_idx];
        }

        char arg_str[] = "MCRG"; // Matrix, Coordinate, Real, General

        mm_write_mtx_crd(&file_name[0], n_rows, n_cols, nnz, &(I)[0], &(J)[0],
                         &(values)[0], arg_str);

        // Revert increments so MatrixCOO object state is consistent if used
        // again
        for (int nz_idx = 0; nz_idx < nnz; ++nz_idx) {
            --I[nz_idx];
            --J[nz_idx];
        }
    }

    void read_from_mtx(const std::string &matrix_file_name) {
#ifdef DEBUG_MODE
        std::cout << "Reading matrix from file: " << matrix_file_name
                  << std::endl;
#endif
#ifdef USE_FAST_MMIO
        std::vector<int> original_rows;
        std::vector<int> original_cols;
        std::vector<double> original_vals;

        fmm::matrix_market_header header;

        // Load
        {
            fmm::read_options options;
            options.generalize_symmetry = true;
            std::ifstream f(matrix_file_name);
            fmm::read_matrix_market_triplet(f, header, original_rows,
                                            original_cols, original_vals,
                                            options);
        }

        // Find sort permutation
        auto perm = compute_sort_permutation(original_rows, original_cols);

        // Apply permutation
        this->I = apply_permutation(perm, original_rows);
        this->J = apply_permutation(perm, original_cols);
        this->values = apply_permutation(perm, original_vals);

        this->n_rows = header.nrows;
        this->n_cols = header.ncols;
        this->nnz = this->values.size();
        this->is_sorted = true;
        this->is_symmetric = (header.symmetry != fmm::symmetry_type::general);
#else
        MM_typecode matcode;
        FILE *f = fopen(matrix_file_name.c_str(), "r");
        if (!f) {
            throw std::runtime_error("Unable to open file: " +
                                     matrix_file_name);
        }

        if (mm_read_banner(f, &matcode) != 0) {
            fclose(f);
            throw std::runtime_error(
                "Could not process Matrix Market banner in file: " +
                matrix_file_name);
        }

        fclose(f);

        if (!(mm_is_sparse(matcode) &&
              (mm_is_real(matcode) || mm_is_pattern(matcode) ||
               mm_is_integer(matcode)) &&
              (mm_is_symmetric(matcode) || mm_is_general(matcode)))) {
            throw std::runtime_error("Unsupported matrix format in file: " +
                                     matrix_file_name);
        }

        int nrows, ncols, nnz;
        int *row_unsorted = nullptr;
        int *col_unsorted = nullptr;
        double *val_unsorted = nullptr;

        if (mm_read_unsymmetric_sparse<double, int>(
                matrix_file_name.c_str(), &nrows, &ncols, &nnz, &val_unsorted,
                &row_unsorted, &col_unsorted) < 0) {
            throw std::runtime_error("Error reading matrix from file: " +
                                     matrix_file_name);
        }

        if (nrows != ncols) {
            throw std::runtime_error("Matrix must be square.");
        }

        bool symm_flag = mm_is_symmetric(matcode);

        std::vector<int> row_data, col_data;
        std::vector<double> val_data;

        // Unpacks symmetric matrices
        // TODO: You should be able to work with symmetric matrices!
        if (symm_flag) {
            for (int i = 0; i < nnz; ++i) {
                row_data.push_back(row_unsorted[i]);
                col_data.push_back(col_unsorted[i]);
                val_data.push_back(val_unsorted[i]);
                if (row_unsorted[i] != col_unsorted[i]) {
                    row_data.push_back(col_unsorted[i]);
                    col_data.push_back(row_unsorted[i]);
                    val_data.push_back(val_unsorted[i]);
                }
            }
            free(row_unsorted);
            free(col_unsorted);
            free(val_unsorted);
            nnz = static_cast<unsigned long long>(val_data.size());
        } else {
            row_data.assign(row_unsorted, row_unsorted + nnz);
            col_data.assign(col_unsorted, col_unsorted + nnz);
            val_data.assign(val_unsorted, val_unsorted + nnz);
            free(row_unsorted);
            free(col_unsorted);
            free(val_unsorted);
        }

        std::vector<int> perm(nnz);
        std::iota(perm.begin(), perm.end(), 0);
        sort_perm(row_data.data(), perm.data(), nnz);

        this->I.resize(nnz);
        this->J.resize(nnz);
        this->values.resize(nnz);

        for (int i = 0; i < nnz; ++i) {
            this->I[i] = row_data[perm[i]];
            this->J[i] = col_data[perm[i]];
            this->values[i] = val_data[perm[i]];
        }

        this->n_rows = nrows;
        this->n_cols = ncols;
        this->nnz = nnz;
        this->is_sorted = 1;    // TODO: verify
        this->is_symmetric = 0; // TODO: determine based on matcode?
#endif

#ifdef DEBUG_MODE
        std::cout << "Completed reading matrix from file: " << matrix_file_name
                  << std::endl;
#endif
    }

    void read_from_mtx_old(const std::string matrix_file_name) {
        char *filename = const_cast<char *>(matrix_file_name.c_str());
        int nrows, ncols, nnz_read; // Renamed nnz to nnz_read to avoid
                                    // confusion with member nnz

        MM_typecode matcode;
        FILE *f;

        if ((f = fopen(filename, "r")) == NULL) {
            printf("Unable to open file\n");
            exit(
                EXIT_FAILURE); // Use exit instead of return for critical errors
        }

        if (mm_read_banner(f, &matcode) != 0) {
            printf(
                "mm_read_unsymetric: Could not process Matrix Market banner ");
            printf(" in file [%s]\n", filename);
            fclose(f); // Close file before exiting
            exit(EXIT_FAILURE);
        }

        fclose(f);

        bool compatible_flag =
            (mm_is_sparse(matcode) &&
             (mm_is_real(matcode) || mm_is_pattern(matcode) ||
              mm_is_integer(matcode))) &&
            (mm_is_symmetric(matcode) || mm_is_general(matcode));
        bool symm_flag = mm_is_symmetric(matcode);

        if (!compatible_flag) {
            printf("The matrix market file provided is not supported.\n Reason "
                   ":\n");
            if (!mm_is_sparse(matcode)) {
                printf(" * matrix has to be sparse\n");
            }
            if (!mm_is_real(matcode) && !(mm_is_pattern(matcode))) {
                printf(" * matrix has to be real or pattern\n");
            }
            if (!mm_is_symmetric(matcode) && !mm_is_general(matcode)) {
                printf(" * matrix has to be either general or symmetric\n");
            }
            exit(EXIT_FAILURE);
        }

        int *row_unsorted;
        int *col_unsorted;
        double *val_unsorted;

        // mm_read_unsymmetric_sparse allocates memory with malloc()
        if (mm_read_unsymmetric_sparse<double, int>(
                filename, &nrows, &ncols, &nnz_read, &val_unsorted,
                &row_unsorted, &col_unsorted) < 0) {
            printf("Error in file reading\n");
            exit(EXIT_FAILURE);
        }

        if (nrows != ncols) {
            printf("Matrix not square. Currently only square matrices are "
                   "supported\n");
            free(row_unsorted); // Free malloc'd memory before exiting
            free(col_unsorted);
            free(val_unsorted);
            exit(EXIT_FAILURE);
        }

        // THESE ARE THE CRUCIAL DECLARATIONS: They must be here!
        int *row_unsorted_ptr = row_unsorted;
        int *col_unsorted_ptr = col_unsorted;
        double *val_unsorted_ptr = val_unsorted;
        long current_nnz =
            nnz_read; // Use a mutable nnz here for symmetric case

        bool allocated_with_new_for_general = false;

        // If matrix market file is symmetric; create a general one out of it
        if (symm_flag) {
            int ctr = 0;
            for (int idx = 0; idx < nnz_read; ++idx) {
                ++ctr;
                if (row_unsorted[idx] != col_unsorted[idx]) {
                    ++ctr;
                }
            }

            int new_nnz = ctr;

            // These are allocated with NEW
            int *row_general = new int[new_nnz];
            int *col_general = new int[new_nnz];
            double *val_general = new double[new_nnz];

            int idx_gen = 0;
            for (int idx = 0; idx < nnz_read; ++idx) {
                row_general[idx_gen] = row_unsorted[idx];
                col_general[idx_gen] = col_unsorted[idx];
                val_general[idx_gen] = val_unsorted[idx];
                ++idx_gen;

                if (row_unsorted[idx] != col_unsorted[idx]) {
                    row_general[idx_gen] = col_unsorted[idx];
                    col_general[idx_gen] = row_unsorted[idx];
                    val_general[idx_gen] = val_unsorted[idx];
                    ++idx_gen;
                }
            }

            // Free the original malloc-ed data now that it's copied
            // These are correct because row_unsorted_ptr etc. still point to
            // the malloc'd memory here.
            free(row_unsorted_ptr);
            free(col_unsorted_ptr);
            free(val_unsorted_ptr);

            current_nnz = new_nnz;

            // Now, the temporary pointers point to the NEW-allocated memory
            row_unsorted_ptr = row_general;
            col_unsorted_ptr = col_general;
            val_unsorted_ptr = val_general;

            allocated_with_new_for_general = true;
        }

        // permute the col and val according to row
        int *tmp_perm = new int[current_nnz];

        for (int idx = 0; idx < current_nnz; ++idx) {
            tmp_perm[idx] = idx;
        }

        sort_perm(row_unsorted_ptr, tmp_perm, current_nnz);

        int *col = new int[current_nnz];
        int *row = new int[current_nnz];
        double *val = new double[current_nnz];

        for (int idx = 0; idx < current_nnz; ++idx) {
            col[idx] = col_unsorted_ptr[tmp_perm[idx]];
            val[idx] = val_unsorted_ptr[tmp_perm[idx]];
            row[idx] = row_unsorted_ptr[tmp_perm[idx]];
        }

        delete[] tmp_perm;

        // Deallocate the `row_unsorted_ptr`, `col_unsorted_ptr`,
        // `val_unsorted_ptr` based on how they were *last* allocated (malloc or
        // new[]).
        if (allocated_with_new_for_general) {
            delete[] row_unsorted_ptr;
            delete[] col_unsorted_ptr;
            delete[] val_unsorted_ptr;
        } else {
            free(row_unsorted_ptr);
            free(col_unsorted_ptr);
            free(val_unsorted_ptr);
        }

        this->values = std::vector<double>(val, val + current_nnz);
        this->I = std::vector<int>(row, row + current_nnz);
        this->J = std::vector<int>(col, col + current_nnz);
        this->n_rows = nrows;
        this->n_cols = ncols;
        this->nnz = current_nnz;
        this->is_sorted = 1;
        this->is_symmetric = 0;

        delete[] val;
        delete[] row;
        delete[] col;
    }

    void print(void) {
        std::cout << "n_rows = " << n_rows << std::endl;
        std::cout << "n_cols = " << n_cols << std::endl;
        std::cout << "nnz = " << nnz << std::endl;
        std::cout << "is_sorted = " << is_sorted << std::endl;
        std::cout << "is_symmetric = " << is_symmetric << std::endl;

        std::cout << "I = [";
        for (int i = 0; i < nnz; ++i) {
            std::cout << I[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "J = [";
        for (int i = 0; i < nnz; ++i) {
            std::cout << J[i] << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "values = [";
        for (int i = 0; i < nnz; ++i) {
            std::cout << values[i] << ", ";
        }
        std::cout << "]" << std::endl;
    }

    void scamac_make_mtx(const std::string matrix_file_name);
};
#ifdef USE_SCAMAC
/* helper function:
 * split integer range [a...b-1] in n nearly equally sized pieces [ia...ib-1],
 * for i=0,...,n-1 */
void split_range(ScamacIdx a, ScamacIdx b, ScamacIdx n, ScamacIdx i,
                 ScamacIdx *ia, ScamacIdx *ib) {
    ScamacIdx m = (b - a - 1) / n + 1;
    ScamacIdx d = n - (n * m - (b - a));
    if (i < d) {
        *ia = m * i + a;
        *ib = m * (i + 1) + a;
    } else {
        *ia = m * d + (i - d) * (m - 1) + a;
        *ib = m * d + (i - d + 1) * (m - 1) + a;
    }
}

void scamac_generate(const std::string matrix_file_name, int *scamac_nrows,
                     int *scamac_nnz, MatrixCOO *coo_mat) {

    /**  examples/MPI/ex_count_mpi.c
     *
     *   basic example:
     *   - read a matrix name/argument string from the command line
     *   - count the number of non-zeros, and compute the maximum norm (=max
     * |entry|) and row-sum norm
     *
     *   Matrix rows are generated in parallel MPI processes.
     *   The ScamacGenerator and ScamacWorkspace is allocated per process.
     */

    const char *matargstr = matrix_file_name.c_str();

    ScamacErrorCode err;
    ScamacGenerator *my_gen;
    char *errstr = NULL;

    // set error handler for MPI (the only global ScaMaC variable!)
    //   scamac_error_handler = my_mpi_error_handler;

    /* parse matrix name & parameters from command line to obtain a
     * ScamacGenerator ... */
    /* an identical generator is created per MPI process */
    err = scamac_parse_argstr(matargstr, &my_gen, &errstr);
    /* ... and check for errors */
    if (err) {
        printf("-- Problem with matrix-argument string:\n-- %s\n---> Abort.\n",
               errstr);
        // my_mpi_error_handler();
    }

    /* check matrix parameters */
    err = scamac_generator_check(my_gen, &errstr);
    if (err) {
        printf("-- Problem with matrix parameters:\n-- %s---> Abort.\n",
               errstr);
        // my_mpi_error_handler();
    }

    /* finalize the generator ... */
    err = scamac_generator_finalize(my_gen);
    /* ... and check, whether the matrix dimension is too large */
    if (err == SCAMAC_EOVERFLOW) {
        // TODO: doesn't work with llvm
        // printf("-- matrix dimension exceeds max. IDX value
        // (%"SCAMACPRIDX")\n---> Abort.\n",SCAMAC_IDX_MAX);
        // my_mpi_error_handler();
    }
    /* catch remaining errors */
    SCAMAC_CHKERR(err);

    /* query number of rows and max. number of non-zero entries per row */
    ScamacIdx nrow = scamac_generator_query_nrow(my_gen);
    ScamacIdx maxnzrow = scamac_generator_query_maxnzrow(my_gen);

    //   double t1 = MPI_Wtime();

    /* ScamacWorkspace is allocated per MPI process */
    ScamacWorkspace *my_ws;
    SCAMAC_TRY(scamac_workspace_alloc(my_gen, &my_ws));

    /* allocate memory for column indices and values per MPI process*/
    //   ScamacIdx *cind = malloc(maxnzrow * sizeof(long int));
    ScamacIdx *cind = new signed long int[maxnzrow];
    double *val;
    if (scamac_generator_query_valtype(my_gen) == SCAMAC_VAL_REAL) {
        // val = malloc(maxnzrow * sizeof *val);
        val = new double[maxnzrow];
    } else {
        /* valtype == SCAMAC_VAL_COMPLEX */
        // val = malloc(2*maxnzrow * sizeof(double));
        val = new double[maxnzrow];
    }

    ScamacIdx ia, ib;
    // this MPI process generates rows ia ... ib-1
    split_range(0, nrow, 1, 0, &ia, &ib);

    // allocate space
    int *scamac_rowPtr = new int[nrow + 1];
    int *scamac_col = new int[maxnzrow * nrow];
    double *scamac_val = new double[maxnzrow * nrow];

    // init counters
    int row_ptr_idx = 0;
    int scs_arr_idx = 0;
    scamac_rowPtr[0] = 0;

    for (ScamacIdx idx = ia; idx < ib; idx++) {
        ScamacIdx k;
        /* generate single row ... */
        SCAMAC_TRY(scamac_generate_row(my_gen, my_ws, idx, SCAMAC_DEFAULT, &k,
                                       cind, val));
        /* ... which has 0 <=k <= maxnzrow entries */

        // Assign SCAMAC arrays to scs array
        scamac_rowPtr[row_ptr_idx + 1] = scamac_rowPtr[row_ptr_idx] + k;
        for (int i = 0; i < k; ++i) {
            scamac_col[scs_arr_idx] =
                cind[i]; // I dont know if these are "remade" every iteration,
                         // seems like it
            scamac_val[scs_arr_idx] = val[i];
            ++scs_arr_idx;
        }

        *scamac_nnz += k;
        ++row_ptr_idx;
    }
    *scamac_nrows = ib - ia;

    // Stupid to convert back to COO, only to convert back to scs. But safe for
    // now.
    (coo_mat->I).resize(*scamac_nnz);
    (coo_mat->J).resize(*scamac_nnz);
    (coo_mat->values).resize(*scamac_nnz);

    // for (int i = 0; i < *scamac_nrows + 1; ++i){
    //     std::cout << "scamac row ptr[" << i << "] = " << scamac_rowPtr[i] <<
    //     std::endl;
    // }

    int elem_num = 0;
    for (int row = 0; row < *scamac_nrows; ++row) {
        for (int idx = scamac_rowPtr[row]; idx < scamac_rowPtr[row + 1];
             ++idx) {
            (coo_mat->I)[elem_num] = row;
            (coo_mat->J)[elem_num] = scamac_col[idx];
            (coo_mat->values)[elem_num] = scamac_val[idx];
            ++elem_num;
        }
    }

    /* free local objects */
    delete[] scamac_rowPtr;
    delete[] scamac_col;
    delete[] scamac_val;

    free(cind);
    free(val);
    SCAMAC_TRY(scamac_workspace_free(my_ws));
    SCAMAC_TRY(scamac_generator_destroy(my_gen));
}

void MatrixCOO::scamac_make_mtx(const std::string matrix_file_name) {
    int scamac_nrows = 0;
    int scamac_nnz = 0;

    // Fill scs arrays with proper data
    scamac_generate(matrix_file_name, &scamac_nrows, &scamac_nnz, this);

    // Finish up mtx struct creation (TODO: why do I do it this way?)
    this->n_rows = (std::set<int>((this->I).begin(), (this->I).end())).size();
    this->n_cols = (std::set<int>((this->J).begin(), (this->J).end())).size();
    this->nnz = (this->values).size();
};
#endif
