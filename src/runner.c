#include "cpu/kernels.h"

#include "utils.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define N 1024
#define K 1024

#define SAMPLE_SIZE 50

static void run_cpu(int kernel_id) {
  srand(420);

  double total_gflops = 0.;
  for (int i = 0; i < SAMPLE_SIZE; i++) {
    float *A = randn(M * K);
    float *B = randn(K * N);
    float *C = zeros(M * N);

    uint64_t start = nanos();
    cpu_kernels[kernel_id](M, N, K, 1.0, A, B, 0.0, C);
    uint64_t end = nanos();

    /* 
      alpha*A@B + beta*C FLOPs = M*N (alpha mul) + M*N*2*K (matmul) + M*N (beta mul) + M*N (addition)
    */
    double flops =  M*N*2.0*K + 3.0*M*N;
    double gflop = flops * 1e-9;
    double seconds = (end - start) * 1e-9;
    double gflops = gflop / seconds;
    printf("%3d: GFLOP/s %.2f\n", i, gflops);
    total_gflops += gflops;

    float *expected = zeros(M * N);
    cpu_kernels[CPU_OPENBLAS](M, N, K, 1.0, A, B, 0.0, expected);
    check_correct(expected, C, M * N);
  }

  printf("---\nMean GFLOP/s %.2f\n", total_gflops / SAMPLE_SIZE);
}

static void print_usage() {
  fprintf(stderr, "usage: gemm <kernel_id: numeric>\n");
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    print_usage();
    return EXIT_FAILURE;
  }

  int kernel_id = atoi(argv[1]);
  if (kernel_id < CPU_START || kernel_id > CPU_END) {
    return EXIT_FAILURE;
  }

  run_cpu(kernel_id);

  return EXIT_SUCCESS;
}
