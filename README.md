# gemm
Step by step matrix multiplication optimization for educational purposes.


## CPU

### 1. Prereqs
Requires [`OpenBlas`](https://github.com/OpenMathLib/OpenBLAS) to be installed in order to check CPU kernels corectness and compare number of FLOPS against.

Make sure to have it installed somewhere where `pkg-config` can access it.

e.g.
```bash
git clone https://github.com/OpenMathLib/OpenBLAS 
cd OpenBLAS                                   
make -C .                                    
make -C . PREFIX=/usr/local install          
```

### 2. Build
```bash
make build
```

If non standard `OpenBlas` installation path, build with:
```bash
OPENBLAS_PATH=<path> make build
```

### 3. Run
```
make run
```

