# gemm
Step by step matrix multiplication optimization for educational purposes.

## CPU

### 1. Naive 
Naive 3-loop iterative implementation.

*Not the most naive*, as it does calculate the dot-product for an element `C[i, j]` in a variable (which likely is stored in a single register) and doesn't read memory at each `C[i, j]` update in the iterative summing.

### 2. Cache friendly loop reordering

Data is transferred between memory and cache in blocks of fixed size, called cache lines or cache blocks. When a cache line is copied from memory into the cache, a cache entry is created. The cache entry will include the copied data as well as the requested memory location (called a tag).

The usual size of a cache line is `64` bytes.

On a high level, when the processor needs to read or write a location in memory, it first checks for a corresponding entry in the cache. The cache checks for the contents of the requested memory location in any cache lines that might contain that address. If the processor finds that the memory location is in the cache, a cache hit has occurred. However, if the processor does not find the memory location in the cache, a cache miss has occurred. In the case of a cache hit, the processor immediately reads or writes the data in the cache line. For a cache miss, the cache allocates a new entry and copies data from main memory, then the request is fulfilled from the contents of the cache.

As rough estimations, an L1 cache reference takes ~1ns, whereas main memory reference takes ~100ns.

When calculating the value of element `C[i, j]` we calculate the dot-product between `A[i, :]` (i-th row of A) and `B[:, j]` (j-th column of B).

Given the strided representation of matrices in memory, and the fact that C is a row-major programming language, it means that the inner loop (to `K`) from the Naive implementation is very cache unfriendly.
For matrix `A`, `K` represents columns (i.e. contiguous memory), but for matrix `B` it represents rows.
When iterating over matrix B (going down the rows), the memory read always cache-misses (assuming B is large enough, i.e. roughly B's row stride > cache line).

We can re-order the loops and swap the `K` loop with the `N` loop. `N` loop indexes into the columns of matrix `B` and we would benefit a lot, cache wise, if that would be done most frequently. 
Now the innermost loop computes partial results, hence we cannot perform accumulation in a single register anymore.

It seems that after this optimization, if compiling with `-march=native` the compiler uses AVX/FMA (`vmovups`, `vfmadd`, etc) instructions and registers (`ymm` - holds 8 single precision floats). 

> objdump -d build/obj/cpu/kernels/loop_reorder.o
```assembly
...
vmovups -224(%r12,%r15,4), %ymm3
vmovups -192(%r12,%r15,4), %ymm4         
vmovups -160(%r12,%r15,4), %ymm5         
vmovups -128(%r12,%r15,4), %ymm6
vfmadd213ps     -224(%rdx,%r15,4), %ymm2, %ymm3 ## ymm3 = (ymm2 * ymm3) + mem
vfmadd213ps     -192(%rdx,%r15,4), %ymm2, %ymm4 ## ymm4 = (ymm2 * ymm4) + mem
vfmadd213ps     -160(%rdx,%r15,4), %ymm2, %ymm5 ## ymm5 = (ymm2 * ymm5) + mem
vfmadd213ps     -128(%rdx,%r15,4), %ymm2, %ymm6 ## ymm6 = (ymm2 * ymm6) + mem
vmovups %ymm3, -224(%rdx,%r15,4)                                             
vmovups %ymm4, -192(%rdx,%r15,4)                                             
vmovups %ymm5, -160(%rdx,%r15,4)                                             
vmovups %ymm6, -128(%rdx,%r15,4)
...
```

### Results [1024x1024 matrices] [Single thread] 
| **Kernel** 	                       | **GFLOPs/s** | **Speed-up over naive**     |
|------------------------------------|:------------:|:---------------------------:|
| [1] Naive      	                   |`1.08`        |`1.0x`                       |
| [2] Cache friendly loop reordering |`22.97`       |`21.2x`                      |
| [3] 1-D Tiling | | |
| [4] 2-D Tiling | | |
| [5] FMA instructions | | |
| [0] OpenBLAS   	                   |`102.23`      |`94.6x`                      |


>Executed on an 2,2 GHz 6-Core Intel Core i7 MacBook Pro.
> 
>Compiled with `FAST=1` i.e. `-O2 -ffast-math -march=native -funroll-loops`. 
>

### How to replicate?
#### 1. Prereqs
Requires [`OpenBlas`](https://github.com/OpenMathLib/OpenBLAS) to be installed in order to check CPU kernels corectness and compare number of FLOPS against.

Make sure to have it installed somewhere where `pkg-config` can access it.

e.g.
```bash
git clone https://github.com/OpenMathLib/OpenBLAS 
cd OpenBLAS                                   
make -C .                                    
make -C . PREFIX=/usr/local install          
```

#### 2. Build
```bash
make build
```

If non standard `OpenBlas` installation path: 
```bash
OPENBLAS_PATH=<path> make build
```

#### 3. Run
```
make run
```

## References
Extremely well written articles which I highly recommend reading:
1. https://marek.ai/matrix-multiplication-on-cpu.html
2. https://siboehm.com/articles/22/Fast-MMM-on-CPU
3. https://siboehm.com/articles/22/CUDA-MMM
