# gtx-1650-tflops-performance-benchmark
In an effort to create a nanogpt clone optimized for GTX 1650, DeepSeek discovered that FP32 performs faster than FP16, despite theoretical performance claims on TechPowerUp and other websites indicating the opposite. Therefore, I requested a script to examine any GPU for these differences.
RESULTS:
========================================================================================
GPU dtype benchmark
========================================================================================
GPU:                 NVIDIA GeForce GTX 1650
Compute capability:  7.5
VRAM:                4.29 GB
SMs:                 14
PyTorch:             2.10.0+cu126
CUDA runtime:        12.6
Driver:              617.14
BF16 supported:      True (may be emulated on this GPU)
TF32 supported:      False
FP16 reduced-prec:   True
========================================================================================

Benchmarking raw GEMM... (iters=10, warmup=3)

=== raw GEMM (CUDA events) ===
dtype          n          ms      TFLOPS   vs FP32     rel.err
--------------------------------------------------------------
FP32        1024        0.97       2.212     1.00x     0.0e+00
FP16        1024        8.61       0.250     0.11x     3.6e-04
BF16        1024        1.69       1.273     0.58x     2.9e-03
--------------------------------------------------------------
FP32        2048        6.79       2.531     1.00x     0.0e+00
FP16        2048       50.40       0.341     0.13x     3.6e-04
BF16        2048       11.14       1.543     0.61x     2.9e-03
--------------------------------------------------------------
FP32        4096       81.74       1.681     1.00x     0.0e+00
FP16        4096      373.93       0.368     0.22x     3.6e-04
BF16        4096       86.50       1.589     0.94x     2.9e-03
--------------------------------------------------------------

GPT benchmark: 30.94M params, T=512, checkpoint=True
Using batch=16, seq_len=512 (8,192 tokens/step)

-- model: fp32 --
   5,947 tok/s | 1.25 GB | loss 9.34

-- model: fp16_autocast --
   1,441 tok/s | 1.50 GB | loss 9.43

-- model: fp16_native --
   1,466 tok/s | 0.97 GB | loss -

-- model: bf16_autocast --
   4,793 tok/s | 1.50 GB | loss 8.96

-- model: bf16_native --
   5,027 tok/s | 0.97 GB | loss 8.95

=== GPT training step ===
mode                     tok/s   vs FP32   peak GB      loss    status
----------------------------------------------------------------------
fp32                     5,947     1.00x      1.25     9.339        ok
fp16_autocast            1,441     0.24x      1.50     9.428        ok
fp16_native              1,466     0.25x      0.97         - nonfinite
bf16_autocast            4,793     0.81x      1.50     8.962        ok
bf16_native              5,027     0.85x      0.97     8.945        ok
----------------------------------------------------------------------
nonfinite/unstable = NaN, inf, or a loss far above random init; throughput is shown but the mode is excluded from the recommendation.
fp16_native is diagnostic (whole model in half, no FP32 master weights/GradScaler), not a recommended training setup.

=== recommendation ===
Fastest training mode: fp32 (5,947 tok/s, 1.00x FP32).
FP16 autocast is slower than FP32 for this workload (0.24x FP32).
BF16 autocast: 0.81x FP32.
Raw GEMM at n=4096: FP32 1.681 TFLOPS, FP16 0.368 TFLOPS (0.22x).
Keep FP32 on this GPU; it is the fastest measured training mode here.
This verdict is specific to the GPU, driver, PyTorch/CUDA build, and shapes above. Re-run after driver or framework updates.
