#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#include "../types.h"

enum CpuKernels {
    CPU_OPENBLAS = 0,
    CPU_START = CPU_OPENBLAS,
    CPU_NAIVE,
    CPU_LOOP_REORDER,
    CPU_1D_TILING,
    CPU_END = CPU_1D_TILING
};

void sgemm_openblas(int m, int n, int k, float alpha, const float *A,
                    const float *B, float beta, float *C);

void sgemm_naive(int m, int n, int k, float alpha, const float *A,
                 const float *B, float beta, float *C);

void sgemm_loop_reorder(int m, int n, int k, float alpha, const float *A,
                        const float *B, float beta, float *C);

void sgemm_1d_tiling(int m, int n, int k, float alpha, const float *A,
                     const float *B, float beta, float *C);

sgemm_kernel cpu_kernels[] = {sgemm_openblas, sgemm_naive, sgemm_loop_reorder, sgemm_1d_tiling};

#endif /* GEMM_KERNELS_H */
