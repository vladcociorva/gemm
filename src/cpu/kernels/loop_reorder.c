void sgemm_loop_reorder(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {

  if (beta != 1.0) {
    for (int y = 0; y < M; y++) {
      for (int x = 0; x < N; x++) {
        C[y * M + x] *= beta;
      }
    }
  }

  for (int y = 0; y < M; y++) {
    for (int i = 0; i < K; i++) {
      float scaledA = alpha * A[y * M + i];
      for (int x = 0; x < N; x++) {
        C[y * M + x] += scaledA * B[i * K + x];
      }
    }
  }
}
