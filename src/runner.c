#include "cpu/kernels.h"

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define N 1024 
#define RUN 1

uint64_t nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
}

float random_sp() {
    return (float)rand() / (float)(RAND_MAX + 1.0f);
}

float *random_sp_matrix_host(uint64_t n) {
    float *mat = (float *)malloc(sizeof(float) * n); 
    for (size_t i = 0; i < n; i++) {
        mat[i] = random_sp();
    }
    return mat;
}

float *zeros_matrix_host(uint64_t n) {
    float *mat = (float *)malloc(sizeof(float) * n); 
    memset(mat, 0, sizeof(float) * n);
    return mat;
}

void print(float *data, uint64_t n, uint64_t m) {
    for (size_t y = 0; y < n; y++) {
        for (size_t x = 0; x < m; x++) {
            printf("%.3f ", data[y * n + x]);
        }
        printf("\n");
    }
    printf("\n");
} 

int main(void) {
    srand(420);

    float *A = random_sp_matrix_host(N*N);
    float *B = random_sp_matrix_host(N*N);
    float *C = zeros_matrix_host(N*N);

    uint64_t start = nanos();
    for (int i = 0; i < RUN; i++) {
        sgemm_naive(N, N, N, 1.0, A, B, 0.0, C);
    }
    uint64_t end = nanos();

    double flops = 2.0*N*N*N * RUN;
    double tflops = flops * 1e-12;
    double seconds = (end - start) * 1e-9;

    printf("TFLOP/S %.5f\n", tflops/seconds);
}
