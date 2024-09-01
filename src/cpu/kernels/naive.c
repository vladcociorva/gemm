void sgemm_naive(int m, int n, int k, float alpha, const float *A,
                 const float *B, float beta, float *C) {
  float tmp = 0.0;
  for (int y = 0; y < m; y++) {
    for (int x = 0; x < n; x++) {
      tmp = 0.0;
      for (int i = 0; i < k; i++) {
        tmp += A[y * k + i] * B[i * n + x];
      }

      C[y * n + x] = alpha * tmp + beta * C[y * n + x];
    }
  }
}
