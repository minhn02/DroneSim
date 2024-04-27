// Quadrotor hovering example
// Make sure in glob_opts.hpp:
// - NSTATES = 12, NINPUTS=4
// - NHORIZON = anything you want
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)
// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://roboticexplorationlab.org/papers/planning_with_attitude.pdf

#include <stdio.h>

#include <tinympc/admm.hpp>
#include "matlib_cpu.h"
#include "tinympc/glob_opts.hpp"
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"
#include "tinympc/variables.hpp"

extern "C"
{

void tracking(float observations[12], float inputs[4])
{
    // Map array from problem_data (array in row-major order)
    tiny::rho = rho_value;
    set((tinytype*)tiny::Kinf, Kinf_data, NINPUTS, NSTATES);
    transpose((tinytype*)tiny::Kinf, (tinytype*)tiny::KinfT, NINPUTS, NSTATES);    
    set((tinytype*)tiny::Pinf, Pinf_data, NSTATES, NSTATES);
    transpose((tinytype*)tiny::Pinf, (tinytype*)tiny::PinfT, NSTATES, NSTATES);
    set((tinytype*)tiny::Quu_inv, Quu_inv_data, NINPUTS, NINPUTS);
    set((tinytype*)tiny::AmBKt, AmBKt_data, NSTATES, NSTATES);
    set((tinytype*)tiny::coeff_d2p, coeff_d2p_data, NSTATES, NINPUTS);
    set((tinytype*)tiny::Adyn, Adyn_data, NSTATES, NSTATES);
    transpose((tinytype*)tiny::Adyn, (tinytype*)tiny::AdynT, NSTATES, NSTATES);
    set((tinytype*)tiny::Bdyn, Bdyn_data, NSTATES, NINPUTS);
    transpose((tinytype*)tiny::Bdyn, (tinytype*)tiny::BdynT, NSTATES, NINPUTS);
    set((tinytype*)tiny::Q, Q_data, NSTATES, 1);
    set((tinytype*)tiny::R, R_data, NSTATES, 1);

    // Valid range for inputs and states
    set((tinytype*)tiny::u_min, -0.583, NINPUTS, NHORIZON-1);
    set((tinytype*)tiny::u_max, 1 - 0.583, NINPUTS, NHORIZON-1);
    set((tinytype*)tiny::x_min, -5, NSTATES, NHORIZON);
    set((tinytype*)tiny::x_max, 5, NSTATES, NHORIZON);

    // Optimization states, inputs, and settings
    tiny::primal_residual_state = 0;
    tiny::primal_residual_input = 0;
    tiny::dual_residual_state = 0;
    tiny::dual_residual_input = 0;
    tiny::status = 0;
    tiny::iter = 0;
    tiny::abs_pri_tol = 0.001;
    tiny::abs_dua_tol = 0.001;
    tiny::max_iter = 100;
    tiny::check_termination = 1;
    tiny::en_input_bound = 1;
    tiny::en_state_bound = 1;


    tinytype v1[NSTATES];
    tinytype v2[NSTATES];

    tinytype Xref_total[NSTATES][NTOTAL];

    set((tinytype*)Xref_total, Xref_data, NSTATES, NTOTAL);
    set((tinytype*)tiny::Xref, (tinytype*)Xref_total, NSTATES, NHORIZON);

    // current and next simulation states
    float x0[NSTATES];
    set(x0, observations, NSTATES, 1);

    for (int k = 0; k < 10; ++k) {

        // Print states array to CSV file
        // calculate the value of (x0 - tiny::Xref.col(1)).norm()
        float xref_col[NSTATES];
        get_col((tinytype*)tiny::Xref, xref_col, 1, NSTATES, NHORIZON);
        matsub(x0, xref_col, v1, 1, NSTATES);
        float norm = matnorm(v1, 1, NSTATES);
        // printf("Tracking error: %0.7f\n", norm);

        // 1. Update measurement
        // an alternative method is to use tiny::x.setCol(x0.data[0], 0);
        set_col((tinytype*)tiny::x, x0, 0, NSTATES, NHORIZON);

        // 2. Update reference (if needed)
        set((tinytype*)tiny::Xref, (tinytype*) (Xref_total + k * NSTATES), NSTATES, NHORIZON);

        // 3. Reset dual variables (if needed)
        set((tinytype*)tiny::y, 0.0, NINPUTS, NHORIZON-1);
        set((tinytype*)tiny::g, 0.0, NSTATES, NHORIZON);

        // 4. Solve MPC problem
        tiny_solve();
        float u_col_0[NINPUTS];
        get_col((tinytype*)tiny::u, u_col_0, 0, NINPUTS, NHORIZON-1);
        for (int i = 0; i < NINPUTS; i++) {
            inputs[i] = u_col_0[i];
        }
        return;
    }
}

int main() {
    float observations[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float inputs[4];
    tracking(observations, inputs);
    printf("inputs: %f, %f, %f, %f\n", inputs[0], inputs[1], inputs[2], inputs[3]);
}

} /* extern "C" */