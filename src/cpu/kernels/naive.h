#ifndef GEMM_CPU_NAIVE_H
#define GEMM_CPU_NAIVE_H

// TODO: implement the scalar part 
void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                 const float *B, float beta, float *C) {
    
    float tmp = 0.0;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            tmp = 0.0;
            for (int i = 0; i < K; i++) {
                tmp += A[y * M + i] * B[i * K + x]; 
            }
            
            C[y * M + x] = tmp;
        }
    }
}

#endif /* GEMM_CPU_NAIVE_H */
