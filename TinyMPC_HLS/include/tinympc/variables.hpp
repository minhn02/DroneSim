#pragma once
#ifndef OPT_VARIABLES_H
#define OPT_VARIABLES_H

#include "glob_opts.hpp"

namespace tiny {

// TinyCache
extern tinytype rho;
extern tinytype Kinf[NSTATES][NINPUTS];
extern tinytype KinfT[NINPUTS][NSTATES];
extern tinytype Pinf[NSTATES][NSTATES];
extern tinytype PinfT[NSTATES][NSTATES];
extern tinytype Quu_inv[NINPUTS][NINPUTS];
extern tinytype AmBKt[NSTATES][NSTATES];
extern tinytype coeff_d2p[NINPUTS][NSTATES];

// TinySetting
extern tinytype abs_pri_tol;
extern tinytype abs_dua_tol;
extern int max_iter;
extern int check_termination;
extern int en_state_bound;
extern int en_input_bound;

// TinyWorkspace

// State and input
extern tinytype x[NHORIZON][NSTATES];
extern tinytype u[NHORIZON-1][NINPUTS];

// Linear control cost terms
extern tinytype q[NHORIZON][NSTATES];
extern tinytype r[NHORIZON-1][NINPUTS];

// Linear Riccati backward pass terms
extern tinytype p[NHORIZON][NSTATES];
extern tinytype d[NHORIZON-1][NINPUTS];

// Auxiliary variables
extern tinytype v[NHORIZON][NSTATES];
extern tinytype vnew[NHORIZON][NSTATES];
extern tinytype z[NHORIZON-1][NINPUTS];
extern tinytype znew[NHORIZON-1][NINPUTS];

// Dual variables
extern tinytype g[NHORIZON][NSTATES];
extern tinytype y[NHORIZON-1][NINPUTS];

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
extern tinytype Bdyn[NINPUTS][NSTATES];
extern tinytype BdynT[NSTATES][NINPUTS];

extern tinytype u_min[NHORIZON-1][NINPUTS];
extern tinytype u_max[NHORIZON-1][NINPUTS];
extern tinytype x_min[NHORIZON][NSTATES];
extern tinytype x_max[NHORIZON][NSTATES];
extern tinytype Xref[NHORIZON][NSTATES];   // Nh x Nx
extern tinytype Uref[NHORIZON-1][NINPUTS]; // Nh-1 x Nu

// Temporaries
extern tinytype Qu[NINPUTS];
extern tinytype u1[NINPUTS];
extern tinytype u2[NINPUTS];
extern tinytype x1[NSTATES];
extern tinytype x2[NSTATES];
extern tinytype x3[NSTATES];
extern tinytype m1[NHORIZON-1][NINPUTS];
extern tinytype m2[NHORIZON-1][NINPUTS];
extern tinytype m3[NHORIZON-1][NINPUTS];
extern tinytype s1[NHORIZON][NSTATES];
extern tinytype s2[NHORIZON][NSTATES];

}

// Nice Functions
inline void get_col(tinytype* in, tinytype *out, int col, int rows, int cols) {    
    for (int i = 0; i < rows; i++) {
        out[i] = in[col * rows + i];
    }
}

inline void set_col(tinytype* a, tinytype* data, int col, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        a[col * rows + i] = data[i];
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