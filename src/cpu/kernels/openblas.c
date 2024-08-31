#include <cblas.h>

void sgemm_openblas(int m, int n, int k, float alpha, const float *A,
                    const float *B, float beta, float *C) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, n,
              B, k, beta, C, k);
}
