#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#include "../types.h"

enum CpuKernels {
    CPU_OPENBLAS = 0,
    CPU_START = CPU_OPENBLAS,
    CPU_NAIVE,
    CPU_LOOP_REORDER,
    CPU_END = CPU_LOOP_REORDER
};

void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                 const float *B, float beta, float *C);

void sgemm_loop_reorder(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C);

void sgemm_openblas(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C);

sgemm_kernel cpu_kernels[] = {sgemm_openblas, sgemm_naive, sgemm_loop_reorder};

#endif /* GEMM_KERNELS_H */
