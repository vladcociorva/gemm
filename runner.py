#!/usr/bin/env python3
import ctypes
import time
import numpy as np

lib = ctypes.CDLL('build/libgemm.so')

def _init_c_sgemm(fn_name: str):
    fn = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.c_int,                    # M
        ctypes.c_int,                    # N
        ctypes.c_int,                    # K
        ctypes.c_float,                  # alpha
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.c_float,                  # beta
        ctypes.POINTER(ctypes.c_float)   # C
    ]
    fn.restype = None
    return fn

c_sgemms = {
    "naive": _init_c_sgemm("sgemm_naive"),
    "open_blas": _init_c_sgemm("sgemm_openblas")
}

def run(sgemm_fn: str, M: int, N: int, K: int, alpha: float, beta: float):
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    A = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C = C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    start = time.monotonic()
    c_sgemms[sgemm_fn](M, N, K, alpha, A, B, beta, C)
    end = time.monotonic()

    gflops = M*N*2*K * 1e-9
    print(f"GFLOPs/S: {gflops/(end-start):.2f}")

    assert np.allclose(A_np @ B_np, C_np, atol=1e-4), "WRONG"


if __name__ == '__main__':
    M, N, K = 1024, 1024, 1024
    alpha, beta = 1.0, 1.0

    for i in range(10000):
        run("naive", M, N, K, alpha, beta)
