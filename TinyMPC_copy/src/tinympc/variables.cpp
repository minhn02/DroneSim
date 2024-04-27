#include "variables.hpp"

namespace tiny {

// TinyCache
tinytype rho;
tinytype Kinf[NINPUTS][NHORIZON-1];
tinytype KinfT[NHORIZON-1][NINPUTS];
tinytype Pinf[NSTATES][NSTATES];
tinytype PinfT[NSTATES][NSTATES];
tinytype Quu_inv[NINPUTS][NINPUTS];
tinytype AmBKt[NSTATES][NSTATES];
tinytype coeff_d2p[NSTATES][NINPUTS];

// TinySetting
tinytype abs_pri_tol;
tinytype abs_dua_tol;
int max_iter;
int check_termination;
int en_state_bound;
int en_input_bound;

// TinyWorkspace

// State and input
tinytype x[NSTATES][NHORIZON];
tinytype u[NINPUTS][NHORIZON-1];

// Linear control cost terms
tinytype q[NSTATES][NHORIZON];
tinytype r[NINPUTS][NHORIZON-1];

// Linear Riccati backward pass terms
tinytype p[NSTATES][NHORIZON];
tinytype d[NINPUTS][NHORIZON-1];

// Auxiliary variables
tinytype v[NSTATES][NHORIZON];
tinytype vnew[NSTATES][NHORIZON];
tinytype z[NINPUTS][NHORIZON-1];
tinytype znew[NINPUTS][NHORIZON-1];

// Dual variables
tinytype g[NSTATES][NHORIZON];
tinytype y[NINPUTS][NHORIZON-1];

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
tinytype Bdyn[NINPUTS][NINPUTS];
tinytype BdynT[NINPUTS][NINPUTS];

tinytype u_min[NINPUTS][NHORIZON-1];
tinytype u_max[NINPUTS][NHORIZON-1];
tinytype x_min[NSTATES][NHORIZON];
tinytype x_max[NSTATES][NHORIZON];
tinytype Xref[NSTATES][NHORIZON];   // Nx x Nh
tinytype Uref[NINPUTS][NHORIZON-1]; // Nu x Nh-1

// Temporaries
tinytype Qu[NINPUTS];
tinytype u1[NINPUTS];
tinytype u2[NINPUTS];
tinytype x1[NSTATES];
tinytype x2[NSTATES];
tinytype x3[NSTATES];
tinytype m1[NINPUTS][NHORIZON-1];
tinytype m2[NINPUTS][NHORIZON-1];
tinytype m3[NINPUTS][NHORIZON-1];
tinytype s1[NSTATES][NHORIZON];
tinytype s2[NSTATES][NHORIZON];

}
