#define T 64
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

void sgemm_1d_tiling(int m, int n, int k, float alpha, const float *A,
                     const float *B, float beta, float *C) {

  if (beta != 1.0) {
    for (int y = 0; y < m; y++) {
      for (int x = 0; x < n; x++) {
        C[y * n + x] *= beta;
      }
    }
  }

  for (int it = 0; it < n; it += T) {
    int tile_end = MIN(k, it + T);
    for (int y = 0; y < m; y++) {
      for (int i = it; i < tile_end; i++) {
        float scaled_A = alpha * A[y * k + i];
        for (int x = 0; x < n; x++) {
          C[y * n + x] += scaled_A * B[i * n + x];
        }
      }
    }
  }
}
