#pragma once
#ifndef TINYMPC_TYPES_H
#define TINYMPC_TYPES_H

#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "glob_opts.hpp"
#include "matlib/matlib.h"

#ifdef RVV_DEFAULT_TO_ROW_MAJOR
#define RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION RowMajor
#else
#define RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ColMajor
#endif

enum StorageOptions {
    /** Storage order is column major (see \ref TopicStorageOrders). */
    ColMajor = 0,
    /** Storage order is row major (see \ref TopicStorageOrders). */
    RowMajor = 0x1,  // it is only a coincidence that this is equal to RowMajorBit -- don't rely on that
    /** Align the matrix itself if it is vectorizable fixed-size */
    AutoAlign = 0,
    DontAlign = 0x2
};

// Forward declarations
template<typename Scalar_, int Rows_, int Cols_,
        int Options_ = AutoAlign |
                       ( (Cols_ == 1 && Rows_ > 1) ? ColMajor
                       : (Rows_ == 1 && Cols_ > 1) ? RowMajor
                       : RVV_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
        int MaxRows_ = Rows_,
        int MaxCols_ = Cols_
> class Matrix {

public:
    Scalar_ _data[MaxRows_ * MaxCols_];
    float *data;
    int rows, cols, outer, inner;

    void _Matrix(int rows_, int cols_) {
        rows = rows_;
        cols = cols_;
        if (Options_ & RowMajor) {
            outer = rows_;
            inner = cols_;
        } else {
            outer = cols_;
            inner = rows_;
        }
        data = _data;
    }

    // Constructor
    Matrix() {
        _Matrix(Rows_, Cols_);
        for (int i = 0; i < outer * inner; ++i) {
            data[i] = 0;
        }
    }

    // Copy Constructor
    Matrix(Matrix& other) {
        assert(other.rows <= MaxRows_ && other.cols <= MaxCols_);
        _Matrix(Rows_, Cols_);
        matcopy(other.data, data, outer, inner);
    }

    // Copy Constructor
    Matrix(Scalar_ *data) {
        _Matrix(Rows_, Cols_);
        matsetv(this->data, data, outer, inner);
    }

    // Column if ColMajor
    void col(int col, Scalar_ *column) {
        assert(!(Options_ & RowMajor));
        
        for (int i = 0; i < rows; i++) {
            column[i] = data[i * cols + col];
        }
    }

    void set_col(int col, Scalar_* column) {
        for (int i = 0; i < rows; i++) {
            data[i * cols + col] = column[i];
        }
    }

    // Assignment Operator
    // TODO: it has a bug in the last statement
    // virtual Matrix& operator=(const Matrix *other) {
    //     if (this == other) return *this;
    //     matcopy(other->data, data, outer, inner);
    //     return *this;
    // }

    // // Assignment Operator
    // virtual Matrix& operator=(const Scalar_ f) {
    //     matset(data, f, outer, inner);
    //     return *this;
    // }

    Matrix& assign(const Matrix *other) {
        if (this == other) return *this;
        matcopy(other->data, data, outer, inner);
        return *this;
    }

    Matrix& assign (const Scalar_ f) {
        matset(data, f, outer, inner);
        return *this;
    }

    // Assignment Operator
    Matrix& set(Scalar_ *f) {
        matsetv(data, f, outer, inner);
        return *this;
    }

    // Access Operator
    Scalar_& operator()(int row, int col) {
        return data[rows*row + col];
    }

    // virtual void toString() {
    //     printf("const array: %x rows: %d cols: %d outer: %d inner: %d (%d, %d)\n", data, rows, cols, outer, inner, Rows_, Cols_);
    // }
};

typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;
typedef Matrix<tinytype, NINPUTS, 1> tiny_VectorNu;
typedef Matrix<tinytype, NSTATES, NSTATES, RowMajor> tiny_MatrixNxNx;
typedef Matrix<tinytype, NSTATES, NINPUTS, RowMajor> tiny_MatrixNxNu;
typedef Matrix<tinytype, NINPUTS, NSTATES, RowMajor> tiny_MatrixNuNx;
typedef Matrix<tinytype, NINPUTS, NINPUTS, RowMajor> tiny_MatrixNuNu;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;       // Nx x Nh
typedef Matrix<tinytype, NINPUTS, NHORIZON - 1> tiny_MatrixNuNhm1; // Nu x Nh-1

/**
 * Matrices that must be recomputed with changes in time step, rho
 */
typedef struct
{
    tinytype rho;
    tiny_MatrixNuNx Kinf;
    tiny_MatrixNxNu KinfT;
    tinytype * Kinf_data;
    tiny_MatrixNxNx Pinf;
    tiny_MatrixNxNx PinfT;
    tinytype * Pinf_data;
    tiny_MatrixNuNu Quu_inv;
    tinytype * Quu_inv_data;
    tiny_MatrixNxNx AmBKt;
    tinytype * AmBKt_data;
    tiny_MatrixNxNu coeff_d2p;
} TinyCache;

/**
 * User settings
 */
typedef struct
{
    tinytype abs_pri_tol;
    tinytype abs_dua_tol;
    int max_iter;
    int check_termination;
    int en_state_bound;
    int en_input_bound;
} TinySettings;

/**
 * Problem variables
 */
typedef struct
{
    // State and input
    tiny_MatrixNxNh x;
    tiny_MatrixNuNhm1 u;

    // Linear control cost terms
    tiny_MatrixNxNh q;
    tiny_MatrixNuNhm1 r;

    // Linear Riccati backward pass terms
    tiny_MatrixNxNh p;
    tiny_MatrixNuNhm1 d;

    // Auxiliary variables
    tiny_MatrixNxNh v;
    tiny_MatrixNxNh vnew;
    tiny_MatrixNuNhm1 z;
    tiny_MatrixNuNhm1 znew;

    // Dual variables
    tiny_MatrixNxNh g;
    tiny_MatrixNuNhm1 y;

    tinytype primal_residual_state;
    tinytype primal_residual_input;
    tinytype dual_residual_state;
    tinytype dual_residual_input;
    int status;
    int iter;

    tiny_VectorNx Q;
    tiny_VectorNx Qf;
    tiny_VectorNu R;
    tiny_MatrixNxNx Adyn;
    tiny_MatrixNxNx AdynT;
    tinytype * Adyn_data;
    tiny_MatrixNxNu Bdyn;
    tiny_MatrixNuNx BdynT;
    tinytype * Bdyn_data;

    tiny_MatrixNuNhm1 u_min;
    tiny_MatrixNuNhm1 u_max;
    tiny_MatrixNxNh x_min;
    tiny_MatrixNxNh x_max;
    tiny_MatrixNxNh Xref;   // Nx x Nh
    tiny_MatrixNuNhm1 Uref; // Nu x Nh-1

    // Temporaries
    tiny_VectorNu Qu;
    tiny_VectorNu u1, u2;
    tiny_VectorNx x1, x2, x3;
    tiny_MatrixNuNhm1 m1, m2;
    tiny_MatrixNxNh s1, s2;
} TinyWorkspace;

/**
 * Main TinyMPC solver structure that holds all information.
 */
typedef struct
{
    TinySettings settings; // Problem settings
    TinyCache cache;       // Problem cache
    TinyWorkspace work;    // Solver workspace
} TinySolver;

#endif
