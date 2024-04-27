//
// Created by widyadewi on 2/23/24.
//

#pragma once
#ifndef TINYMPC_ADMM_HPP
#define TINYMPC_ADMM_HPP

#include "tinympc/variables.hpp"

#ifndef USE_MATVEC
#define USE_MATVEC 1
#endif

void tiny_init();
int tiny_solve();
void update_primal();
void backward_pass();
void forward_pass();
void update_slack();
void update_dual();
void update_linear_cost();

#endif //TINYMPC_ADMM_HPP
