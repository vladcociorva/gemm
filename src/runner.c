#include "cpu/kernels.h"

#include "utils.h"

#include <bits/getopt_core.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_SIZE 50

static void run_cpu(int kernel_id, int m, int n, int k) {
  srand(420);

  double total_gflops = 0.;
  for (int i = 0; i < SAMPLE_SIZE; i++) {
    float *A = randn(m * k);
    float *B = randn(k * n);
    float *C = zeros(m * n);

    uint64_t start = nanos();
    cpu_kernels[kernel_id](m, n, k, 1.5, A, B, 0.0, C);
    uint64_t end = nanos();

    /*
      alpha*A@B + beta*C FLOPs = M*N (alpha mul) + M*N*2*K (matmul)
                                + M*N (beta mul) + M*N (addition)
    */
    double flops = m * n * 2.0 * k + 3.0 * m * n;
    double gflop = flops * 1e-9;
    double seconds = (end - start) * 1e-9;
    double gflops = gflop / seconds;
    printf("%3d: GFLOP/s %.2f\n", i, gflops);
    total_gflops += gflops;

    float *expected = zeros(m * n);
    cpu_kernels[CPU_OPENBLAS](m, n, k, 1.5, A, B, 0.0, expected);
    check_correct(expected, C, m * n);
  }

  printf("---\nMean GFLOP/s %.2f\n", total_gflops / SAMPLE_SIZE);
}

static void print_usage() {
  fprintf(stderr,
          "Uses kernel i to multiply random matrices A and B of dims m x k and "
          "k x n. Runs %d times while printing the GFLOP/s.\n"
          "e.g., gemm -i 2 -m 2048 -n 2048 -k 2048\n",
          SAMPLE_SIZE);
}

int main(int argc, char *argv[]) {

  int kernel_id, m, n, k;
  kernel_id = m = n = k = -1;

  int opt;
  while ((opt = getopt(argc, argv, "i:m:n:k:")) != -1) {
    switch (opt) {
    case 'i':
      kernel_id = atoi(optarg);
      break;
    case 'm':
      m = atoi(optarg);
      break;
    case 'n':
      n = atoi(optarg);
      break;
    case 'k':
      k = atoi(optarg);
      break;
    default:
      print_usage();
      return EXIT_FAILURE;
    }
  }

  if (kernel_id == -1 || m == -1 || n == -1 || k == -1) {
    print_usage();
    return EXIT_FAILURE;
  }

  run_cpu(kernel_id, m, n, k);

  return EXIT_SUCCESS;
}
