
#include <stdio.h>
#include <math.h>

// // Golden (reference) function declarations
float maxcoeff_cpu(float a[], int n, int m);
float mincoeff_cpu(float a[], int n, int m);
void matneg_cpu(float a[], float b[], int n, int m);
void cwiseabs_cpu(float a[], float b[], int n, int m);
void cwisemin_cpu(float a[], float b[], float c[], int n, int mr);
void cwisemax_cpu(float a[], float b[], float c[], int n, int m);
void cwisemul_cpu(float a[], float b[], float c[], int n, int m);
void matmul_cpu(float a[], float b[], float c[], int n, int m, int o);
void matvec_cpu(float a[], float b[], float c[], int n, int m);
void matvec_transpose_cpu(float a[], float b[], float c[], int n, int m);
void matmulf_cpu(float a[], float b[], float f, int n, int m);
void matsub_cpu(float a[], float b[], float c[], int n, int m);
void matadd_cpu(float a[], float b[], float c[], int n, int m);
void transpose_cpu(float a[], float b[], int n, int m);
void matcopy_cpu(const float a[], float b[], int n, int m);
void matset_cpu(float a[], float f, int n, int m);

#define matsetv matsetv_cpu
#define maxcoeff maxcoeff_cpu
#define mincoeff mincoeff_cpu
#define matnorm matnorm_cpu
#define matneg matneg_cpu
#define cwiseabs cwiseabs_cpu
#define cwisemin cwisemin_cpu
#define cwisemax cwisemax_cpu
#define cwisemul cwisemul_cpu
#define matmul matmul_cpu
#define matvec matvec_cpu
#define matvec_transpose matvec_transpose_cpu
#define matmulf matmulf_cpu
#define matsub matsub_cpu
#define matadd matadd_cpu
#define transpose transpose_cpu
#define matcopy matcopy_cpu
#define matset matset_cpu

float matnorm_cpu(float a[], int n, int m);

float maxcoeff_cpu(float a[], int n, int m);

void matvec_cpu(float a[], float b[], float c[], int n, int m);

float matlib_cpu_top(float a[], int n, int m);

void matsetv_cpu(float a[], float f[], int n, int m);