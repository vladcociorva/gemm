#include <cblas.h>

void sgemm_openblas(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, N,
              B, K, beta, C, K);
}
