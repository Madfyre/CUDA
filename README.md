# CUDA Kernels

A personal collection of CUDA / C++ GPU kernels I implemented while working through
an intensive CUDA course (Yandex School of Data Analysis, infrastructure track).
Each folder under [`kernels/`](kernels/) contains a single self-contained kernel and a
short `NOTES.md` describing what it does and the techniques it uses.

> These are my own implementations. The course's automated grading harness and test
> suite have been removed; what remains is the kernel code itself, meant to be read as
> implementation samples.

## Highlights

The kernels most relevant to modern ML-systems work (low-precision compute, GEMM,
LLM inference primitives):

| Kernel | What it does | Techniques |
|---|---|---|
| [`04-gemm-v1`](kernels/04-gemm-v1/) | FP16 tiled GEMM (RowMajor × ColMajor → ColMajor) | Shared-memory tiling with bank-conflict padding, 1D register blocking (8 outputs/thread), FP32 accumulation, full boundary handling |
| [`09-cutlass-gemm`](kernels/09-cutlass-gemm/) | FP16 TN HGEMM with a fused `HardSwish` epilogue | CUTLASS device API, tensor-core path, custom fused element-wise epilogue |
| [`05-quantization`](kernels/05-quantization/) | INT8 weight quantization for a linear layer | Per-column bias/scale balancing, round-to-nearest, max-based dynamic scale |
| [`06-moe-topk`](kernels/06-moe-topk/) | Mixture-of-Experts Top-K router (FP16) | Per-token Top-K with stable lowest-index tie-breaking (Mixtral-style routing) |
| [`08-rms-norm-gated`](kernels/08-rms-norm-gated/) | RMSNormGated block from Qwen3-Next (FP16) | Fused gated RMS normalization, power-of-two head sizes |
| [`08-prefix-sum`](kernels/08-prefix-sum/) | Work-efficient exclusive scan (N up to 1e8) | Multi-level blocked scan with cross-block carry propagation |

## Full index

**GEMM & tensor cores** — [`02-gemm-v0`](kernels/02-gemm-v0/) (baseline tiled),
[`04-gemm-v1`](kernels/04-gemm-v1/) (FP16 + register blocking),
[`09-cutlass-gemm`](kernels/09-cutlass-gemm/) (CUTLASS + fused epilogue)

**Low-precision & LLM primitives** — [`05-quantization`](kernels/05-quantization/) (INT8),
[`06-moe-topk`](kernels/06-moe-topk/) & [`07-moe-topk-hist`](kernels/07-moe-topk-hist/) (MoE routing),
[`08-rms-norm-gated`](kernels/08-rms-norm-gated/) (Qwen3-Next RMSNormGated)

**Parallel primitives** — [`03-dot-product`](kernels/03-dot-product/) (reduction),
[`08-prefix-sum`](kernels/08-prefix-sum/) (scan),
[`03-transpose-v0`](kernels/03-transpose-v0/) & [`04-transpose-v1`](kernels/04-transpose-v1/) (transpose)

**Other** — [`01-grayscale`](kernels/01-grayscale/) (image), [`05-quaternions`](kernels/05-quaternions/) (quaternion math)

**Warm-ups** — [`00-hello-world`](kernels/00-hello-world/), [`01-device-add`](kernels/01-device-add/),
[`01-gpu-puzzles-1`](kernels/01-gpu-puzzles-1/), [`01-reverse-string`](kernels/01-reverse-string/)

## Building

The kernels target the CUDA Toolkit (tested with a recent `nvcc`). Most are header-only
kernels plus a small shared helper in [`common/`](common/). A single kernel can be compiled
against your own driver, e.g.:

```bash
nvcc -std=c++17 -O3 -arch=sm_80 -Icommon -Ikernels/04-gemm-v1 \
     your_driver.cu -o gemm
```

`09-cutlass-gemm` additionally requires [CUTLASS](https://github.com/NVIDIA/cutlass)
on the include path.

## Benchmarks

*(To add: measured throughput vs. cuBLAS/cuDNN baselines on <your GPU>, e.g. FP16 GEMM
and INT8 quantization. Numbers go here once measured.)*

## License

MIT — see [LICENSE](LICENSE).
