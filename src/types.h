#ifndef GEMM_TYPES_H
#define GEMM_TYPES_H

typedef void (*sgemm_kernel)(int, int, int, float, const float *, const float *, float, float *);

#endif /* GEMM_TYPES_H */
