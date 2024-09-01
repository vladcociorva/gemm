void sgemm_loop_reorder(int m, int n, int k, float alpha, const float *A,
                        const float *B, float beta, float *C) {

  if (beta != 1.0) {
    for (int y = 0; y < m; y++) {
      for (int x = 0; x < n; x++) {
        C[y * n + x] *= beta;
      }
    }
  }

  for (int y = 0; y < m; y++) {
    for (int i = 0; i < k; i++) {
      float scaledA = alpha * A[y * k + i];
      for (int x = 0; x < n; x++) {
        C[y * n + x] += scaledA * B[i * n + x];
      }
    }
  }
}
