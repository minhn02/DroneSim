#include "hls_math.h"
#include "matlib_cpu.h"

// testbenches assisted by chatgpt
int matnorm_tb(){
    // Testbench for Matnorm computes norm of all elements
    float arr[] = {0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0};
    int n = 3;
    int m = 3;
    float ret_res = matnorm_cpu(arr, n, m);

    // Compare results
    float exp_res = 6.0;
    int retval = exp_res == ret_res ? 0 : 1;
    printf("Expected return value: %f\nReal return value: %f\n", exp_res, ret_res);
    if (ret_res != exp_res) {
        printf("Test failed");
    } else {
        printf("Test passed");
    }
    return retval;
}
int matvec_tb() {
    // Testbench inputs
    float a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Example matrix 'a'
    float b[] = {1.0, 2.0};                      // Example vector 'b'
    int n = 3;                                    // Number of rows
    int m = 2;                                    // Number of columns

    // Compute result using the function
    float c[3]; // Result vector 'c'
    matvec_cpu(a, b, c, n, m);

    // Compare results
    float exp_res[] = {5.0, 11.0, 17.0}; // Expected result for the given inputs
    bool passed = true;
    for (int i = 0; i < n; ++i) {
        if (c[i] != exp_res[i]) {
            passed = false;
            break;
        }
    }

    // Print test result
    std::cout << "Expected result: [5.0, 11.0, 17.0]" << std::endl;
    std::cout << "Computed result: [";
    for (int i = 0; i < n; ++i) {
        std::cout << c[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    if (passed) {
        std::cout << "Test passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test failed" << std::endl;
        return 1;
    }
}
int mincoeff_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns
    const int size = n * m;

    // Deterministic input matrix
    float a[size] = {1.2, 2.3, 3.4,
                     4.5, 5.6, 6.7,
                     7.8, 8.9, 9.0};

    // Expected result for the given input matrix
    float expected_result = 1.2;

    // Call HLS function
    float result;
    result= mincoeff_cpu(a, n, m);

    // Compare results
    std::cout << "Expected result: " << expected_result << std::endl;
    std::cout << "Computed result: " << result << std::endl;

    if (expected_result == result) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matnev_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns
    const int size = n * m;

    // Input matrix
    float a[size] = {1.2, -2.3, 3.4,
                     -4.5, 5.6, -6.7,
                     7.8, -8.9, 9.0};

    // Expected result for the given input matrix
    float expected_result[size] = {-1.2, 2.3, -3.4,
                                    4.5, -5.6, 6.7,
                                    -7.8, 8.9, -9.0};

    // Output matrix
    float b[size];

    // Call HLS function
    matneg_cpu(a, b, n, m);

    // Compare results
    bool passed = true;
    for (int i = 0; i < size; ++i) {
        if (b[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int cwiseabs_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns
    const int size = n * m;

    // Input matrix
    float a[size] = {1.2, -2.3, 3.4,
                     -4.5, 5.6, -6.7,
                     7.8, -8.9, 9.0};

    // Expected result for the given input matrix
    float expected_result[size] = {1.2, 2.3, 3.4,
                                    4.5, 5.6, 6.7,
                                    7.8, 8.9, 9.0};

    // Output matrix
    float b[size];

    // Call HLS function
    cwiseabs_cpu(a, b, n, m);

    // Compare results
    bool passed = true;
    for (int i = 0; i < size; ++i) {
        if (b[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
// Testbench
int cwisemin_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns
    const int size = n * m;

    // Input matrices
    float a[size] = {1.2, -2.3, 3.4,
                     -4.5, 5.6, -6.7,
                     7.8, -8.9, 9.0};
    float b[size] = {0.5, -1.0, 2.0,
                     -3.0, 4.0, -5.0,
                     6.0, -7.0, 8.0};

    // Expected result for the given input matrices
    float expected_result[size] = {0.5, -2.3, 2.0,
                                    -4.5, 4.0, -6.7,
                                    6.0, -8.9, 8.0};

    // Output matrix
    float c[size];

    // Call HLS function
    cwisemin_cpu(a, b, c, n, m);

    // Compare results
    bool passed = true;
    for (int i = 0; i < size; ++i) {
        if (c[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int cwisemul_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns
    const int size = n * m;

    // Input matrices
    float a[size] = {1.2, -2.3, 3.4,
                     -4.5, 5.6, -6.7,
                     8.0, -9.0, 9.0};
    float b[size] = {0.5, -1.0, 2.0,
                     -3.0, 4.0, -5.0,
                     6.0, -7.0, 8.0};

    // Expected result for the given input matrices
    float expected_result[size] = {0.6, 2.3, 6.8,
                                    13.5, 22.4, 33.5,
                                    48.0, 63.0, 72.0};

    // Output matrix
    float c[size];

    // Call HLS function
    cwisemul_cpu(a, b, c, n, m);

    // Compare results
    bool passed = true;
    for (int i = 0; i < size; ++i) {
        if (fabs(c[i] - expected_result[i]) > 1e-6) {
            std::cout << "i:" << i << "\tc[i]:" << c[i] << "\texpected[i]:" << expected_result[i] << "!=";
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int cwisemax_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns
    const int size = n * m;

    // Input matrices
    float a[size] = {1.2, -2.3, 3.4,
                     -4.5, 5.6, -6.7,
                     7.8, -8.9, 9.0};
    float b[size] = {0.5, -1.0, 2.0,
                     -3.0, 4.0, -5.0,
                     6.0, -7.0, 8.0};

    // Expected result for the given input matrices
    float expected_result[size] = {1.2, -1.0, 3.4,
                                    -3.0, 5.6, -5.0,
                                    7.8, -7.0, 9.0};

    // Output matrix
    float c[size];

    // Call HLS function
    cwisemax_cpu(a, b, c, n, m);

    // Compare results
    bool passed = true;
    for (int i = 0; i < size; ++i) {
        if (c[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matmul_cpu_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns
    const int o = 4; 

    // Input matrices
    float a[n * o] = {1.0, 2.0, 3.0, 4.0,
                      5.0, 6.0, 7.0, 8.0,
                      9.0, 10.0, 11.0, 12.0};

    float b[m * o] = {0.5, 1.0, 1.5, 2.0,
                      2.5, 3.0, 3.5, 4.0,
                      4.5, 5.0, 5.5, 6.0};

    // Expected result for the given input matrices
    float expected_result[n * m] = {35, 40, 45, 79, 92, 105, 123, 144, 165};

    // Output matrix
    float c[n * m];

    // Call HLS function
    matmul(a, b, c, n, m, o);

    // Compare results
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (c[i] != expected_result[i]) {
           std::cout << "i:" << i << "\tc1[i]:" << c[i] << "\texpected[i]:" << expected_result[i] << "!=";
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 0;
    }
}
int matvec_transpose_cpu_tb() {
    const int n = 4; // Number of rows
    const int m = 4; // Number of columns

    // Input matrix a (column-major)
    float a[n * m] = {9, 6, 5, 4,
                      1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 8, 7, 6};

    // Input vector b
    float b[n] = {1, 2, 3, 4};

    // Expected result for the given input matrix and vector
    float expected_result[m] = {70, 100, 118};

    // Output vector c
    float c[m];

    // Call HLS function
    matvec_transpose(a, b, c, n, m);

    // Compare results
    bool passed = true;
    for (int i = 0; i < m; ++i) {
        if (c[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matmulf_tb() {
    const int n = 3; // Number of rows
    const int m = 3; // Number of columns

    // Input matrix
    float a[n * m] = {1, 2, 3,
                      4, 5, 6,
                      7, 8, 9};

    // Scalar multiplier
    float scalar = 2.5;

    // Expected result for the given input matrix and scalar
    float expected_result[n * m] = {2.5, 5.0, 7.5,
                                    10.0, 12.5, 15.0,
                                    17.5, 20.0, 22.5};

    // Output matrix
    float b[n * m];

    // Call HLS function
    matmulf_cpu(a, b, scalar, n, m);

    // Compare results
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (b[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matsub_tb() {
    const int n = 3; // Number of rows
    const int m = 4; // Number of columns

    // Input matrices
    float a[n * m] = {1.0, 2.0, 3.0, 4.0,
                      5.0, 6.0, 7.0, 8.0,
                      9.0, 10.0, 11.0, 12.0};
    float b[n * m] = {0.5, 1.0, 1.5, 2.0,
                      2.5, 3.0, 3.5, 4.0,
                      4.5, 5.0, 5.5, 6.0};

    // Expected result for the given input matrices
    float expected_result[n * m] = {0.5, 1.0, 1.5, 2.0,
                                    2.5, 3.0, 3.5, 4.0,
                                    4.5, 5.0, 5.5, 6.0};

    // Output matrix
    float c[n * m];

    // Call CPU function
    matsub_cpu(a, b, c, n, m);

    // Compare matrices
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (c[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    // Print result
    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int transpose_tb() {
    const int n = 3; // Number of rows
    const int m = 4; // Number of columns

    // Input matrix
    float a[n * m] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12};

    // Expected output matrix
    float expected_b[m * n];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            expected_b[j * n + i] = a[i * m + j];

    // Output matrix
    float b[m * n];

    // Call CPU function
    transpose_cpu(a, b, n, m);

    // Compare matrices
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (b[i] != expected_b[i]) {
            passed = false;
            break;
        }
    }

    // Print result
    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matcopy_tb() {
    const int n = 3; // Number of rows
    const int m = 4; // Number of columns

    // Input matrix
    float a[n * m] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12};

    // Expected output matrix (copy of input)
    float expected_b[n * m];
    for (int i = 0; i < n * m; ++i) {
        expected_b[i] = a[i];
    }

    // Output matrix
    float b[n * m];

    // Call CPU function
    matcopy_cpu(a, b, n, m);

    // Compare matrices
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (b[i] != expected_b[i]) {
            passed = false;
            break;
        }
    }

    // Print result
    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matset_tb() {
    const int n = 3; // Number of rows
    const int m = 4; // Number of columns

    // Scalar value to set in the matrix
    float scalar = 5.0;

    // Expected output matrix (all elements set to scalar)
    float expected_a[n * m];
    for (int i = 0; i < n * m; ++i) {
        expected_a[i] = scalar;
    }

    // Output matrix
    float a[n * m];

    // Call CPU function
    matset_cpu(a, scalar, n, m);

    // Compare matrices
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (a[i] != expected_a[i]) {
            passed = false;
            break;
        }
    }

    // Print result
    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matsetv_tb() {
    const int n = 3; // Number of rows
    const int m = 4; // Number of columns

    // Input array
    float f[n * m] = {1.0, 2.0, 3.0, 4.0,
                      5.0, 6.0, 7.0, 8.0,
                      9.0, 10.0, 11.0, 12.0};

    // Expected output matrix
    float expected_a[n * m] = {1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0};

    // Output matrix
    float a[n * m];

    // Call CPU function
    matsetv_cpu(a, f, n, m);

    // Compare matrices
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (a[i] != expected_a[i]) {
            passed = false;
            break;
        }
    }

    // Print result
    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int matadd_tb() {
    const int n = 3; // Number of rows
    const int m = 4; // Number of columns

    // Input matrices
    float a[n * m] = {1.0, 2.0, 3.0, 4.0,
                      5.0, 6.0, 7.0, 8.0,
                      9.0, 10.0, 11.0, 12.0};
    float b[n * m] = {0.5, 1.0, 1.5, 2.0,
                      2.5, 3.0, 3.5, 4.0,
                      4.5, 5.0, 5.5, 6.0};

    // Expected result for the given input matrices
    float expected_result[n * m] = {1.5, 3.0, 4.5, 6.0,
                                    7.5, 9.0, 10.5, 12.0,
                                    13.5, 15.0, 16.5, 18.0};

    // Output matrix
    float c[n * m];

    // Call CPU function
    matadd_cpu(a, b, c, n, m);

    // Compare matrices
    bool passed = true;
    for (int i = 0; i < n * m; ++i) {
        if (c[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    // Print result
    if (passed) {
        std::cout << "Test Passed" << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed" << std::endl;
        return 1;
    }
}
int main() {
    return transpose_tb();
}

