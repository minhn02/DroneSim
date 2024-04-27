#pragma once
#ifndef OPT_VARIABLES_H
#define OPT_VARIABLES_H

#include "glob_opts.hpp"

namespace tiny {

// TinyCache
extern tinytype rho;
extern tinytype Kinf[NINPUTS][NHORIZON-1];
extern tinytype KinfT[NHORIZON-1][NINPUTS];
extern tinytype Pinf[NSTATES][NSTATES];
extern tinytype PinfT[NSTATES][NSTATES];
extern tinytype Quu_inv[NINPUTS][NINPUTS];
extern tinytype AmBKt[NSTATES][NSTATES];
extern tinytype coeff_d2p[NSTATES][NINPUTS];

// TinySetting
extern tinytype abs_pri_tol;
extern tinytype abs_dua_tol;
extern int max_iter;
extern int check_termination;
extern int en_state_bound;
extern int en_input_bound;

// TinyWorkspace

// State and input
extern tinytype x[NSTATES][NHORIZON];
extern tinytype u[NINPUTS][NHORIZON-1];

// Linear control cost terms
extern tinytype q[NSTATES][NHORIZON];
extern tinytype r[NINPUTS][NHORIZON-1];

// Linear Riccati backward pass terms
extern tinytype p[NSTATES][NHORIZON];
extern tinytype d[NINPUTS][NHORIZON-1];

// Auxiliary variables
extern tinytype v[NSTATES][NHORIZON];
extern tinytype vnew[NSTATES][NHORIZON];
extern tinytype z[NINPUTS][NHORIZON-1];
extern tinytype znew[NINPUTS][NHORIZON-1];

// Dual variables
extern tinytype g[NSTATES][NHORIZON];
extern tinytype y[NINPUTS][NHORIZON-1];

extern tinytype primal_residual_state;
extern tinytype primal_residual_input;
extern tinytype dual_residual_state;
extern tinytype dual_residual_input;
extern int status;
extern int iter;

extern tinytype Q[NSTATES];
extern tinytype Qf[NSTATES];
extern tinytype R[NINPUTS];
extern tinytype Adyn[NSTATES][NSTATES];
extern tinytype AdynT[NSTATES][NSTATES];
extern tinytype Bdyn[NINPUTS][NINPUTS];
extern tinytype BdynT[NINPUTS][NINPUTS];

extern tinytype u_min[NINPUTS][NHORIZON-1];
extern tinytype u_max[NINPUTS][NHORIZON-1];
extern tinytype x_min[NSTATES][NHORIZON];
extern tinytype x_max[NSTATES][NHORIZON];
extern tinytype Xref[NSTATES][NHORIZON];   // Nx x Nh
extern tinytype Uref[NINPUTS][NHORIZON-1]; // Nu x Nh-1

// Temporaries
extern tinytype Qu[NINPUTS];
extern tinytype u1[NINPUTS];
extern tinytype u2[NINPUTS];
extern tinytype x1[NSTATES];
extern tinytype x2[NSTATES];
extern tinytype x3[NSTATES];
extern tinytype m1[NINPUTS][NHORIZON-1];
extern tinytype m2[NINPUTS][NHORIZON-1];
extern tinytype m3[NINPUTS][NHORIZON-1];
extern tinytype s1[NSTATES][NHORIZON];
extern tinytype s2[NSTATES][NHORIZON];

}

// Nice Functions
inline void get_col(tinytype* in, tinytype *out, int col, int rows, int cols) {    
    for (int i = 0; i < rows; i++) {
        out[i] = in[i * cols + col];
    }
}

inline void set_col(tinytype* a, tinytype* data, int col, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        a[i * cols + col] = data[i];
    }
}

inline void set(tinytype* a, tinytype* data, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        a[i] = data[i];
    }
}

inline void set(tinytype* a, tinytype data, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        a[i] = data;
    }
}

#endif