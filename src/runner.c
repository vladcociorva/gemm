#include "cpu/kernels.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1024

uint64_t nanos() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
}

float random_sp() { return (float)rand() / (float)(RAND_MAX + 1.0f); }

float *random_sp_matrix_host(uint64_t size) {
  float *mat = (float *)malloc(sizeof(float) * size);
  for (size_t i = 0; i < size; i++) {
    mat[i] = random_sp();
  }
  return mat;
}

float *zeros_matrix_host(uint64_t size) {
  float *mat = (float *)malloc(sizeof(float) * size);
  memset(mat, 0, sizeof(float) * size);
  return mat;
}

void print(float *data, uint64_t n, uint64_t m) {
  for (size_t y = 0; y < n; y++) {
    for (size_t x = 0; x < m; x++) {
      printf("%3f ", data[y * n + x]);
    }
    printf("\n");
  }
  printf("\n");
}

#define EPS 9e-2
void check_correct(float *expected, float *actual, uint64_t n) {
  for (size_t i = 0; i < n; i++) {
    if (fabsf(expected[i] - actual[i]) > EPS) {
      printf("difference at %zu: %.5f %.5f\n", i, expected[i], actual[i]);
      exit(-1);
    }
  }
}

int main(void) {
  srand(420);

  for (int i = 0; i < 10; i++) {
    float *A = random_sp_matrix_host(N * N);
    float *B = random_sp_matrix_host(N * N);
    float *C = zeros_matrix_host(N * N);

    uint64_t start = nanos();
    sgemm_loop_reorder(N, N, N, 1.5, A, B, 0.0, C);
    uint64_t end = nanos();

    double flops = 2.0 * N * N * N + 3.0 * N * N;
    double gflops = flops * 1e-9;
    double seconds = (end - start) * 1e-9;
    printf("GFLOPs/S %.5f\n", gflops / seconds);

    if (i == 0) {
      float *expected = zeros_matrix_host(N * N);
      sgemm_openblas(N, N, N, 1.5, A, B, 0.0, expected);
      check_correct(expected, C, N * N);
    }
  }
}
