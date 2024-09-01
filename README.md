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

As a rough estimation of order of magnitude, an L1 cache reference takes ~1ns, whereas main memory reference takes ~100ns.

When calculating the value of element $\bm{C}_{i,j}$ we calculate the dot-product between $\bm{A}_{i,:}$ (i-th row of $\bm{A}$) and $\bm{B}_{:,j}$ (j-th column of $\bm{B}$).

Given the strided representation of matrices in memory, and the fact that C is a row-major programming language, it means that the inner loop (to $k$) from the Naive implementation is very cache unfriendly.
For matrix $\bm{A}$, $k$ represents columns (i.e. contiguous memory), but for matrix $\bm{B}$ it represents rows.
When iterating over matrix $\bm{B}$ (going down the rows), the memory read always cache-misses (assuming $\bm{B}$ is large enough, i.e. roughly $\bm{B}$'s row stride > cache line).

We can re-order the loops and swap the $k$ loop with the $n$ loop. $n$ loop indexes into the columns of matrix $\bm{B}$ and we would benefit a lot, cache wise, if that would be done most frequently. 
Now the innermost loop computes partial results, hence we cannot perform accumulation in a single register anymore.

#### O2 vs O3
We see a big performance increase either way, as the loops are now a bit more friendly to the CPU cache.

Interestingly though, on my machine, there is a **HUGE** difference between compiling with `-O2` and `-O3`. The compiler produces very different asm code.

Performance wise, with `O2`, I get an average of `9.54` GFLOP/s from 50 runs, which is **12.8** times faster than the naive version.

With `O3` however, I get an average of `57.82` GFLOP/s!!. This is **78.1** times faster than the naive version and **6** times faster than the same actual code, but compiled with `-O2`.

##### Deep dive into the generated assembly code

> objdump -d build/obj/cpu/kernels/loop_reorder.o 

`O2` seems to use AVX SIMD instructions for the calculations. It doesn't use any FMA (fused multiply-add) instructions.

It only uses `xmm` registers (i.e., 128bit/4 FP32s at a time). So no `ymm` (256bit/8 FP32s) or `zmm` (512bit/16 FP32s).

Uses just the scalar AVX ops (i.e., `vmulss`, `vaddss`, `vmovss`), which only work on a single floating-point of a `xmm` register, so it's not really taking advantage of SIMD parallelism.

```
...
100:   c5 f2 59 21             vmulss (%rcx),%xmm1,%xmm4
104:   48 83 c0 04             add    $0x4,%rax
108:   48 83 c1 04             add    $0x4,%rcx
10c:   c5 da 58 60 fc          vaddss -0x4(%rax),%xmm4,%xmm4
111:   c5 fa 11 60 fc          vmovss %xmm4,-0x4(%rax)
...
```

Also, the `O2` variant doesn't unroll the loops automatically, this can be forced with `-funroll-loops` though. 
Adding `-funroll-loops` to the `O2` variant seems to net us a consistent extra `0.7` GFLOP/s. From ~`9.5` to about `10.2`.

The assembly code generated with `O3` is far more complicated. It looks to have unrolled everything with the idea of taking advantage of SIMD. Interestingly enough, it still doesn't use FMA instructions.

One core part to note is that it seems to do to packed muls (`vmulps`) and adds (`vaddps`) on `zmm` registers. (i.e., does calculations on 16 FP32s pairs in parallel).
```
3a0:   62 d1 6c 48 59 44 15    vmulps 0x0(%r13,%rdx,1),%zmm2,%zmm0  
3a7:   00                                                                         
3a8:   62 f1 7c 48 58 04 10    vaddps (%rax,%rdx,1),%zmm0,%zmm0
3af:   62 f1 7c 48 11 04 10    vmovups %zmm0,(%rax,%rdx,1)     
3b6:   48 83 c2 40             add    $0x40,%rdx     
3ba:   48 39 d3                cmp    %rdx,%rbx                
3bd:   75 e1                   jne    3a0 <sgemm_loop_reorder+0x3a0>
```

### Results

Hardware:
* **CPU** AMD Ryzen 7 9700X 8 Core @ 5.5GHz (at max)
* **RAM**: 64GB DDR5 @ 4.8GHz
* **OS**: Ubuntu 24.04
* **Compiler**: clang 18.1.3
* **Compiler flags** (i.e., `FAST=1 make build`): `-O3 -march=native`

Experiment:
* 1024x1024 normally distributed square matrices
* 50 runs each. Used the mean GFLOPs/s
* Single thread only

| **Kernel** 	                     | **GFLOP/s** | **Speed-up over naive**     |**Performance relative to OpenBLAS**|
|------------------------------------|:------------:|:---------------------------:|:----------------------------------:|
| [1] Naive      	                 |`0.74`        |`1.0x`                       |`0.3%`                              |
| [2] Cache friendly loop reordering |`57.82`       |`78.1x`                      |`34.5%`                             |
| [0] OpenBLAS **  	                 |`167.48`      |`226.3x`                     |`100%`                              |


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

* [**\*\***] Manually limited to one thread by running with `OMP_NUM_THREADS=1`. It's a lot faster (~10x), if we let it utilize all the cores.

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
