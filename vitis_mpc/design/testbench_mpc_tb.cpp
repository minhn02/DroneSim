
#include "include/top_quadrotor.h"
#include <stdio.h>

// testbenches assisted by chatgpt
int tracking_tb(){
    float obs[12] = {0};
    float inputs[4] = {0};
    tracking(obs, inputs);

    printf("inputs: %f %f %f %f", inputs[0], inputs[1], inputs[2], inputs[3]);

    return 0;
}

int main() {
    return tracking_tb();
}

