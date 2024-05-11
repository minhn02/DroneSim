#pragma once
#ifndef TINYMPC_MATLIB_CPU_H
#define TINYMPC_MATLIB_CPU_H

#include <hls_math.h>
#include "matlib_cpu.h"

extern "C"
{

}

// matrix maximum coefficient
float maxcoeff_cpu(float a[], int n, int m) {
    static float max = std::numeric_limits<float>::min();
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS unroll
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS unroll
            #pragma HLS LOOP_TRIPCOUNT max=15 avg=12
            max = a[i * m + j] > max ? a[i * m + j] : max;
        }
    }
    return max;
}

// matrix min coefficient
 float mincoeff_cpu(float a[], int n, int m) {
    static float min = std::numeric_limits<float>::max();
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS UNROLL
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            min = a[i * m + j] < min ? a[i * m + j] : min;
        }
    }
    return min;
}

// matrix unary negative
void matneg_cpu(float a[], float b[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS UNROLL
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            b[i * m + j] = -a[i * m + j];
        }
    }
}

// matrix coefficient-wise abs
void cwiseabs_cpu(float a[], float b[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=12 min=10
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=12 min=10
            #pragma HLS UNROLL 
            b[i * m + j] = fabs(a[i * m + j]);
        }
    }
}

// matrix coefficient-wise min
void cwisemin_cpu(float a[], float b[], float c[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        LOOP_INNER: for (int j = 0; j < m; ++j) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS UNROLL
            c[i * m + j] = a[i * m + j] < b[i * m + j] ? a[i * m + j] : b[i * m + j];
        }
    }
}

// matrix coefficient-wise multiplication
void cwisemul_cpu(float a[], float b[], float c[], int n, int m) {
        LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
                c[i * m + j] = a[i * m + j] * b[i * m + j];
            }
    }
}

// matrix coefficient-wise max
void cwisemax_cpu(float a[], float b[], float c[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            c[i * m + j] = a[i * m + j] > b[i * m + j] ? a[i * m + j] : b[i * m + j];
        }
    }
}

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
void matmul_cpu(float a[], float b[], float c[], int n, int m, int o) {
    LOOP_ROW: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        LOOP_COL: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            c[i * m + j] = 0;
            LOOP_WRITE: for (int k = 0; k < o; ++k) {
                #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
                #pragma HLS UNROLL
                c[i * m + j] += a[i * o + k] * b[k * m + j];
            }
        }
    }

}

/*  a is row major
 *        j
 *    1 2 3 4     9
 *  i 5 6 7 8  *  6
 *    9 8 7 6     5 j
 *                4
 */

void matvec_cpu(float a[], float b[], float c[], int n, int m) {
    OUTER_LOOP: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS UNROLL
        c[i] = 0;
        INNER_LOOP: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            c[i] += a[i * m + j] * b[j];
        }
    }

}


/*  a is col major
 *      j         i
 *  9 6 5 4  *  1 5 9
 *              2 6 8
 *              3 7 7 j
 *              4 8 6
 */

void matvec_transpose_cpu(float a[], float b[], float c[], int n, int m) {
    OUTER_LOOP: for (int i = 0; i < m; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS UNROLL
        c[i] = 0;
        INNER_LOOP: for (int j = 0; j < n; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS PIPELINE
            c[i] += a[j * m + i] * b[j];
        }
    }
}

// matrix scalar multiplication
void matmulf_cpu(float a[], float b[], float f, int n, int m) {
    OUTER_LOOP: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS PIPELINE
        INNER_LOOP: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            b[i * m + j] = f * a[i * m + j];
        }
    }
}

// matrix subtraction
void matsub_cpu(float a[], float b[], float c[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS PIPELINE
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            c[i * m + j] = a[i * m + j] - b[i * m + j];
        }
    }
}

// matrix addition
void matadd_cpu(float a[], float b[], float c[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS PIPELINE
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            c[i * m + j] = a[i * m + j] + b[i * m + j];
        }
    }
}

// matrix transpose
void transpose_cpu(float a[], float b[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS PIPELINE
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            b[j * n + i] = a[i * m + j];
        }
    }
}

// matrix copy
void matcopy_cpu(const float a[], float b[], int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS PIPELINE
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            b[i * m + j] = a[i * m + j];
        }
    }
}

void matset_cpu(float a[], float f, int n, int m) {
    LOOP_OUTER: for (int i = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS PIPELINE
        LOOP_INNER: for (int j = 0; j < m; ++j) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma UNROLL
            a[i * m + j] = f;
        }
    }
}

void matsetv_cpu(float a[], float f[], int n, int m) {
    LOOP_OUTER: for (int i = 0, k = 0; i < n; ++i) {
        #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
        #pragma HLS PIPELINE
        LOOP_INNER: for (int j = 0; j < m; ++j, ++k) {
            #pragma HLS LOOP_TRIPCOUNT max=12 avg=10 min=10
            #pragma HLS UNROLL
            a[i * m + j] = f[k];
        }
    }
}

// matrix l2 norm
float matnorm_cpu(float a[], int n, int m) {
    static float sum = 0;
    for (int i = 0; i < n; ++i) {
        #pragma HLS pipeline 
        for (int j = 0; j < m; ++j) {
            #pragma HLS unroll
            sum += a[i * m + j] * a[i * m + j];
        }
    }
    return hls::sqrt(sum);
}

void matlib_cpu_top_array(float *a, float *b, float *c, int n, int m) {
    // function to be tested, currently not using and setting in config file
    matvec_cpu(a, b, c, n, m);
}

#endif // TINYMPC_MATLIB_CPU_H
