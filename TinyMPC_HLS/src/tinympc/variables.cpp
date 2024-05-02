#include "variables.hpp"
#include "glob_opts.hpp"

namespace tiny {

// TinyCache
tinytype rho;
tinytype Kinf[NSTATES][NINPUTS];
tinytype KinfT[NINPUTS][NSTATES];
tinytype Pinf[NSTATES][NSTATES];
tinytype PinfT[NSTATES][NSTATES];
tinytype Quu_inv[NINPUTS][NINPUTS];
tinytype AmBKt[NSTATES][NSTATES];
tinytype coeff_d2p[NINPUTS][NSTATES];

// TinySetting
tinytype abs_pri_tol;
tinytype abs_dua_tol;
int max_iter;
int check_termination;
int en_state_bound;
int en_input_bound;

// TinyWorkspace

// State and input
tinytype x[NHORIZON][NSTATES];
tinytype u[NHORIZON-1][NINPUTS];

// Linear control cost terms
tinytype q[NHORIZON][NSTATES];
tinytype r[NHORIZON-1][NINPUTS];

// Linear Riccati backward pass terms
tinytype p[NHORIZON][NSTATES];
tinytype d[NHORIZON-1][NINPUTS];

// Auxiliary variables
tinytype v[NHORIZON][NSTATES];
tinytype vnew[NHORIZON][NSTATES];
tinytype z[NHORIZON-1][NINPUTS];
tinytype znew[NHORIZON-1][NINPUTS];

// Dual variables
tinytype g[NHORIZON][NSTATES];
tinytype y[NHORIZON-1][NINPUTS];

tinytype primal_residual_state;
tinytype primal_residual_input;
tinytype dual_residual_state;
tinytype dual_residual_input;
int status;
int iter;

tinytype Q[NSTATES];
tinytype Qf[NSTATES];
tinytype R[NINPUTS];
tinytype Adyn[NSTATES][NSTATES];
tinytype AdynT[NSTATES][NSTATES];
tinytype Bdyn[NINPUTS][NSTATES];
tinytype BdynT[NSTATES][NINPUTS];

tinytype u_min[NHORIZON-1][NINPUTS];
tinytype u_max[NHORIZON-1][NINPUTS];
tinytype x_min[NHORIZON][NSTATES];
tinytype x_max[NHORIZON][NSTATES];
tinytype Xref[NHORIZON][NSTATES];   // Nh x Nx
tinytype Uref[NHORIZON-1][NINPUTS]; // Nh-1 x Nu

// Temporaries
tinytype Qu[NINPUTS];
tinytype u1[NINPUTS];
tinytype u2[NINPUTS];
tinytype x1[NSTATES];
tinytype x2[NSTATES];
tinytype x3[NSTATES];
tinytype m1[NHORIZON-1][NINPUTS];
tinytype m2[NHORIZON-1][NINPUTS];
tinytype m3[NHORIZON-1][NINPUTS];
tinytype s1[NHORIZON][NSTATES];
tinytype s2[NHORIZON][NSTATES];

}

void get_col(tinytype* in, tinytype *out, int col, int rows, int cols) {    
    for (int i = 0; i < rows; i++) {
        #pragma HLS unroll factor=4
        out[i] = in[col * rows + i];
    }
}

void set_col(tinytype* a, tinytype* data, int col, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        #pragma HLS unroll factor=4
        a[col * rows + i] = data[i];
    }
}

void set(tinytype* a, tinytype* data, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        a[i] = data[i];
        if (i == 144) {
            return;
        }
    }
}

void set(tinytype* a, tinytype data, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        #pragma HLS unroll factor=4
        a[i] = data;
    }
}