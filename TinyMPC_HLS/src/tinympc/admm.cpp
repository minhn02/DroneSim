#include "tinympc/admm.hpp"
#include "glob_opts.hpp"
#include "variables.hpp"
#include "matlib.h"

// u1 = x[:, i] * Kinf; u2 = u1 + d; u[:, i] = -u2
void forward_pass_1(int i) {
    tinytype x_col[NSTATES];
    get_col((tinytype*)tiny::x, x_col, i, NSTATES, NHORIZON);
    tinytype d_col[NINPUTS];
    get_col((tinytype*)tiny::d, d_col, i, NINPUTS, NHORIZON-1);
    tinytype u_col[NINPUTS];
    get_col((tinytype*)tiny::u, u_col, i, NINPUTS, NHORIZON-1);

    matvec((tinytype*)tiny::Kinf, x_col, tiny::u1, NINPUTS, NSTATES);
    matadd(tiny::u1, d_col, tiny::u2, 1, NINPUTS);
    matneg(tiny::u2, u_col, 1, NINPUTS);
    set_col((tinytype*)tiny::u, u_col, i, NINPUTS, NHORIZON-1);
}

// x[:, i+1] = Adyn * x[:, i] + Bdyn * u[:, i]
void forward_pass_2(int i) {
    tinytype x_col[NSTATES];
    get_col((tinytype*)tiny::x, x_col, i, NSTATES, NHORIZON);
    tinytype u_col[NINPUTS];
    get_col((tinytype*)tiny::u, u_col, i, NINPUTS, NHORIZON-1);
    tinytype x_col_p1[NSTATES];
    get_col((tinytype*)tiny::x, x_col_p1, i+1, NSTATES, NHORIZON);

    matvec((tinytype*)tiny::Adyn, x_col, tiny::x1, NSTATES, NSTATES);
    matvec((tinytype*)tiny::Bdyn, u_col, tiny::x2, NSTATES, NINPUTS);
    matadd(tiny::x1, tiny::x2, x_col_p1, 1, NSTATES);

    set_col((tinytype*)tiny::x, x_col_p1, i+1, NSTATES, NHORIZON);
}

// d[:, i] = Quu_inv * (BdynT * p[:, i+1] + r[:, i]);
void backward_pass_1(int i) {
    tinytype p_col_p1[NSTATES];
    get_col((tinytype*)tiny::p, p_col_p1, i+1, NSTATES, NHORIZON);
    tinytype r_col[NINPUTS];
    get_col((tinytype*)tiny::r, r_col, i, NINPUTS, NHORIZON-1);
    tinytype d_col[NINPUTS];
    get_col((tinytype*)tiny::d, d_col, i, NINPUTS, NHORIZON-1);

    matvec((tinytype*)tiny::BdynT, p_col_p1, tiny::u1, NINPUTS, NSTATES);
    matadd(r_col, tiny::u1, tiny::u2, 1, NINPUTS);
    matvec((tinytype*)tiny::Quu_inv, tiny::u2, d_col, NINPUTS, NINPUTS);

    set_col((tinytype*)tiny::d, d_col, i, NINPUTS, NHORIZON-1);
}

// p[:, i] = q[:, i] + AmBKt * p[:, i + 1] - KinfT * r[:, i]
void backward_pass_2(int i) {
    tinytype p_col_p1[NSTATES];
    get_col((tinytype*)tiny::p, p_col_p1, i+1, NSTATES, NHORIZON);
    tinytype r_col[NINPUTS];
    get_col((tinytype*)tiny::r, r_col, i, NINPUTS, NHORIZON-1);
    tinytype q_col[NSTATES];
    get_col((tinytype*)tiny::q, q_col, i, NSTATES, NHORIZON);
    tinytype p_col[NSTATES];
    get_col((tinytype*)tiny::p, p_col, i, NSTATES, NHORIZON);

    matvec((tinytype*)tiny::AmBKt, p_col_p1, tiny::x1, NSTATES, NSTATES);
    matvec((tinytype*)tiny::KinfT, r_col, tiny::x2, NSTATES, NINPUTS);
    matsub(tiny::x1, tiny::x2, tiny::x3, 1, NSTATES);
    matadd(tiny::x3, q_col, p_col, 1, NSTATES);

    set_col((tinytype*)tiny::p, p_col, i, NSTATES, NHORIZON);
}

// y u znew  g x vnew
void update_dual_1() {
    matadd((tinytype*)tiny::y, (tinytype*)tiny::u, (tinytype*)tiny::m1, NHORIZON - 1, NINPUTS);
    matsub((tinytype*)tiny::m1, (tinytype*)tiny::znew, (tinytype*)tiny::y, NHORIZON - 1, NINPUTS);
    matadd((tinytype*)tiny::g, (tinytype*)tiny::x, (tinytype*)tiny::s1, NHORIZON, NSTATES);
    matsub((tinytype*)tiny::s1, (tinytype*)tiny::vnew, (tinytype*)tiny::g, NHORIZON, NSTATES);
}

// Box constraints on input
void update_slack_1() {
    matadd((tinytype*)tiny::u, (tinytype*)tiny::y, (tinytype*)tiny::znew, NHORIZON - 1, NINPUTS);
    if (tiny::en_input_bound) {
        cwisemax((tinytype*)tiny::u_min, (tinytype*)tiny::znew, (tinytype*)tiny::m1, NHORIZON - 1, NINPUTS);
        cwisemin((tinytype*)tiny::u_max, (tinytype*)tiny::m1, (tinytype*)tiny::znew, NHORIZON - 1, NINPUTS);
    }
}

// Box constraints on state
void update_slack_2() {
    matadd((tinytype*)tiny::x, (tinytype*)tiny::g, (tinytype*)tiny::vnew, NHORIZON, NSTATES);
    if (tiny::en_state_bound) {
        cwisemax((tinytype*)tiny::x_min, (tinytype*)tiny::vnew, (tinytype*)tiny::s1, NHORIZON, NSTATES);
        cwisemin((tinytype*)tiny::x_max, (tinytype*)tiny::s1, (tinytype*)tiny::vnew, NHORIZON, NSTATES);
    }
}

void primal_residual_state() {
    matsub((tinytype*)tiny::x, (tinytype*)tiny::vnew, (tinytype*)tiny::s1, NHORIZON, NSTATES);
    cwiseabs((tinytype*)tiny::s1, (tinytype*)tiny::s2, NHORIZON, NSTATES);
    tiny::primal_residual_state = maxcoeff((tinytype*)tiny::s2, NHORIZON, NSTATES);
}

void dual_residual_state() {
    matsub((tinytype*)tiny::v, (tinytype*)tiny::vnew, (tinytype*)tiny::s1, NHORIZON, NSTATES);
    cwiseabs((tinytype*)tiny::s1, (tinytype*)tiny::s2, NHORIZON, NSTATES);
    tiny::dual_residual_state = maxcoeff((tinytype*)tiny::s2, NHORIZON, NSTATES) * tiny::rho;
}

void primal_residual_input() {
    matsub((tinytype*)tiny::u, (tinytype*)tiny::znew, (tinytype*)tiny::m1, NHORIZON - 1, NINPUTS);
    cwiseabs((tinytype*)tiny::m1, (tinytype*)tiny::m2, NHORIZON - 1, NINPUTS);
    tiny::primal_residual_input = maxcoeff((tinytype*)tiny::m2, NHORIZON - 1, NINPUTS);
}

void dual_residual_input() {
    matsub((tinytype*)tiny::z, (tinytype*)tiny::znew, (tinytype*)tiny::m1, NHORIZON - 1, NINPUTS);
    cwiseabs((tinytype*)tiny::m1, (tinytype*)tiny::m2, NHORIZON - 1, NINPUTS);
    tiny::dual_residual_input = maxcoeff((tinytype*)tiny::m2, NHORIZON - 1, NINPUTS) * tiny::rho;
}

void update_linear_cost_1() {
    matsub((tinytype*)tiny::znew, (tinytype*)tiny::y, (tinytype*)tiny::m1, NHORIZON - 1, NINPUTS);
    matmulf((tinytype*)tiny::m1, (tinytype*)tiny::r, -tiny::rho, NHORIZON - 1, NINPUTS);
}

void update_linear_cost_2(int i) {
    tinytype xref_col[NSTATES];
    get_col((tinytype*)tiny::Xref, xref_col, i, NSTATES, NHORIZON);
    tinytype q_col[NSTATES];
    get_col((tinytype*)tiny::q, q_col, i, NSTATES, NHORIZON);

    cwisemul(xref_col, (tinytype*)tiny::Q, (tinytype*)tiny::x1, 1, NSTATES);
    matneg((tinytype*)tiny::x1, q_col, 1, NSTATES);

    set_col((tinytype*)tiny::q, q_col, i, NSTATES, NHORIZON);
}

void update_linear_cost_3() {
    matsub((tinytype*)tiny::vnew, (tinytype*)tiny::g, (tinytype*)tiny::s1, NHORIZON, NSTATES);
    matmulf((tinytype*)tiny::s1, (tinytype*)tiny::s2, tiny::rho, NHORIZON, NSTATES);
    matsub((tinytype*)tiny::q, (tinytype*)tiny::s2, (tinytype*)tiny::s1, NHORIZON, NSTATES);

    set((tinytype*)tiny::q, (tinytype*)tiny::s1, NSTATES, NHORIZON);
}

void update_linear_cost_4() {
    tinytype vnew_col_h1[NSTATES];
    get_col((tinytype*)tiny::vnew, vnew_col_h1, NHORIZON-1, NSTATES, NHORIZON);
    tinytype g_col_h1[NSTATES];
    get_col((tinytype*)tiny::g, g_col_h1, NHORIZON-1, NSTATES, NHORIZON);
    tinytype xref_col_h1[NSTATES];
    get_col((tinytype*)tiny::Xref, xref_col_h1, NHORIZON-1, NSTATES, NHORIZON);
    tinytype p_col_h1[NSTATES];
    get_col((tinytype*)tiny::p, p_col_h1, NHORIZON-1, NSTATES, NHORIZON);

    matsub(vnew_col_h1, g_col_h1, (tinytype*)tiny::x1, 1, NSTATES);
    matmulf((tinytype*)tiny::x1, (tinytype*)tiny::x2, tiny::rho, 1, NSTATES);
    matvec((tinytype*)tiny::PinfT, xref_col_h1, (tinytype*)tiny::x1, NSTATES, NSTATES);
    matadd((tinytype*)tiny::x1, (tinytype*)tiny::x2, (tinytype*)tiny::x3, 1, NSTATES);
    matneg((tinytype*)tiny::x3, p_col_h1, 1, NSTATES);

    set_col((tinytype*)tiny::p, p_col_h1, NHORIZON-1, NSTATES, NHORIZON);
}

/**
 * Update linear terms from Riccati backward pass
 */
void backward_pass()
{
    for (int i = NHORIZON - 2; i >= 0; i--) {
        backward_pass_1(i);
        backward_pass_2(i);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
 */
void forward_pass()
{
    for (int i = 0; i < NHORIZON - 1; i++) {
        forward_pass_1(i);
        forward_pass_2(i);
    }
}

/**
 * Do backward Riccati pass then forward roll out
 */
void update_primal()
{
    backward_pass();
    forward_pass();
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint projection function
 */
void update_slack()
{
    update_slack_1();
    update_slack_2();
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
 */
void update_dual()
{
    update_dual_1();
}

/**
 * Update linear control cost terms in the Riccati feedback using
 * the changing slack and dual variables from ADMM
 */
void update_linear_cost()
{
    update_linear_cost_1();
    for (int i = 0; i < NHORIZON; i++) {
        update_linear_cost_2(i);
    }
    update_linear_cost_3();
    update_linear_cost_4();
}

void tiny_init() {
}

int tiny_solve()
{
    // Initialize variables
    tiny::status = 11; // TINY_UNSOLVED
    tiny::iter = 1;

    forward_pass();
    update_slack();
    update_dual();
    update_linear_cost();
    
    
    for (int i = 0; i < tiny::max_iter; i++)
    {
        // Solve linear system with Riccati and roll out to get new trajectory
        update_primal();
        // Project slack variables into feasible domain
        update_slack();
        // Compute next iteration of dual variables
        update_dual();
        // Update linear control cost terms using reference trajectory, duals, and slack variables
        update_linear_cost();


        if (tiny::iter % tiny::check_termination == 0)
        {
            primal_residual_state();
            dual_residual_state();
            primal_residual_input();
            dual_residual_input();

            if (tiny::primal_residual_state < tiny::abs_pri_tol &&
                tiny::primal_residual_input < tiny::abs_pri_tol &&
                tiny::dual_residual_state < tiny::abs_dua_tol &&
                tiny::dual_residual_input < tiny::abs_dua_tol)
            {
                // Solved without error (return 0)
                tiny::status = 1;
                return tiny::status;
            }
        }

        // Save previous slack variables
        set((tinytype*)tiny::v, (tinytype*)tiny::vnew, NSTATES, NHORIZON);
        set((tinytype*)tiny::z, (tinytype*)tiny::znew, NINPUTS, NHORIZON-1);

        tiny::iter += 1;
    }
    return tiny::status;
}
