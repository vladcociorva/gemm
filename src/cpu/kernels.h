#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                 const float *B, float beta, float *C);

void sgemm_loop_reorder(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C);

void sgemm_openblas(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C);

#endif /* GEMM_KERNELS_H */
