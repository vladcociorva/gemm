# gemm
Step by step matrix multiplication optimization for educational purposes.

## CPU

We want to multiply a matrix $\bm{A}$ of shape $m \times k$ with matrix $\bm{B}$ of shape $k \times n$ and get matrix $\bm{C}$ of shape $m \times n$.

### 1. Naive 
Naive 3-loop iterative implementation.

*Not the most naive*, as it does calculate the dot-product for an element $\bm{C}_{i,j}$ in a variable (which likely is stored in a single register) and doesn't read memory at each $\bm{C}_{i,j}$ update in the iterative summing. This makes it such that the CPU cache is not polluted with useles reads for $\bm{C}$ thus havig more space elements of $\bm{A}$ and $\bm{B}$ matrices.

### 2. Cache friendly loop reordering

Data is transferred between memory and cache in blocks of fixed size, called cache lines or cache blocks. When a cache line is copied from memory into the cache, a cache entry is created. The cache entry will include the copied data as well as the requested memory location (called a tag).

The usual size of a cache line is `64` bytes, meaning it will read `16` contiguous fp32s at a time. `getconf -a | grep CACHE` to see exact numbers for cache sizes and cache line sizes. 

On a high level, when the processor needs to read or write a location in memory, it first checks for a corresponding entry in the cache. The cache checks for the contents of the requested memory location in any cache lines that might contain that address. If the processor finds that the memory location is in the cache, a cache hit has occurred. However, if the processor does not find the memory location in the cache, a cache miss has occurred. In the case of a cache hit, the processor immediately reads or writes the data in the cache line. For a cache miss, the cache allocates a new entry and copies data from main memory, then the request is fulfilled from the contents of the cache.

As a rough estimation, an L1 cache reference takes ~1ns, whereas main memory reference takes ~100ns.

When calculating the value of element $\bm{C}_{i,j}$ we calculate the dot-product between $\bm{A}_{i,:}$ (i-th row of $\bm{A}$) and $\bm{B}_{:,j}$ (j-th column of $\bm{B}$).

Given the strided representation of matrices in memory, and the fact that C is a row-major programming language, it means that the inner loop (to $k$) from the Naive implementation is very cache unfriendly.
For matrix $\bm{A}$, $k$ represents columns (i.e. contiguous memory), but for matrix $\bm{B}$ it represents rows.
When iterating over matrix $\bm{B}$ (going down the rows), the memory read always cache-misses (assuming $\bm{B}$ is large enough, i.e. roughly $\bm{B}$'s row stride > cache line).

We can re-order the loops and swap the $k$ loop with the $n$ loop. $n$ loop indexes into the columns of matrix $\bm{B}$ and we would benefit a lot, cache wise, if that would be done most frequently. 
Now the innermost loop computes partial results, hence we cannot perform accumulation in a single register anymore.


### Results [1024x1024 sq matrices] [Single threaded]

Run with 
* CPU AMD Ryzen 7 9700X 8 Core @ 5.5GHz (during load)
* RAM: 64GB DDR5 @ 4.8GHz
* OS: Ubuntu 24.04
* Compiler: clang 18.1.3
* Compiler flags (i.e., `FAST=1 make build`): `-O2 -ffast-math -march=native -funroll-loops`

| **Kernel** 	                     | **GFLOPs/s** | **Speed-up over naive**     |**Performance relative to OpenBLAS**|
|------------------------------------|:------------:|:---------------------------:|:----------------------------------:|
| [1] Naive      	                 |`0.76`        |`1.0x`                       |`0.4%`                             |
| [2] Cache friendly loop reordering*|`10.75`       |`14.1x`                      |`6.6%`                              |
| [0] OpenBLAS **  	                 |`162.61`      |`213.9x`                     |`100%`                              |


* [**\***] It seems that after this optimization, if compiling with `-march=native`, on an old Intel Macbook, the compiler automatically uses AVX/FMA instructions (`vmovups`, `vfmadd`, etc) and registers (`ymm` - which can hold 8 single precision floats at a time). This resulted in about `22.97` GFLOPs/s. Not entirely sure why it doesn't do it on my desktop CPU.
    ```
    > objdump -d build/obj/cpu/kernels/loop_reorder.o
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

* [**\*\***] Manually limited to one thread by running with `OMP_NUM_THREADS=1`

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
./build/gemm <kernel-id>
```

## References
Extremely well written articles which I highly recommend reading:
1. https://marek.ai/matrix-multiplication-on-cpu.html
2. https://siboehm.com/articles/22/Fast-MMM-on-CPU
3. https://siboehm.com/articles/22/CUDA-MMM
