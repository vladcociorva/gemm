
#ifndef GEMM_UTILS_H
#define GEMM_UTILS_H

#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint64_t nanos() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1e9 + ts.tv_nsec;
}

static float random_uniform() { return rand() / (float)RAND_MAX; }

static float random_normal() {
    static int has_spare = 0;
    static float spare;
    float u, v, s;

    if(has_spare) {
        has_spare = 0;
        return spare;
    }

    do {
        u = 2.0f * random_uniform() - 1.0f;
        v = 2.0f * random_uniform() - 1.0f;
        s = u * u + v * v;
    } while(s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = 1;

    return u * s;
}

float *randn(uint64_t size) {
  float *m = (float *)malloc(sizeof(float) * size);
  for (size_t i = 0; i < size; i++) {
    m[i] = random_normal();
  }
  return m;
}


float *zeros(uint64_t size) {
  float *m = (float *)malloc(sizeof(float) * size);
  memset(m, 0, sizeof(float) * size);
  return m;
}

#define EPS 1e-3
void check_correct(float *expected, float *actual, uint64_t n) {
  for (size_t i = 0; i < n; i++) {
    if (fabsf(expected[i] - actual[i]) > EPS) {
      printf("difference at %zu: %.5f %.5f\n", i, expected[i], actual[i]);
      exit(-1);
    }
  }
}

#endif /* GEMM_UTILS_H */