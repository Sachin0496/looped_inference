# Latent decode attention over an FP4 compressed cache: benchmark summary (Kaggle Tesla T4, sm_75)

Source: results/attn_bench_v1/results.json (Kaggle run sachin0496/sensenovakv-attn-bench, version 1, COMPLETE).
Kernel: engine/kernels/kv4_attn.cu. Reference: engine/ref/kv4_attn_ref.py. Bench: kaggle/attn_bench/bench_template.py.

Measured copy bandwidth (256 MB device copy, read+write convention): 241.8 GB/s. All percentages below are of this number.
Timing: all calls for one measurement captured in one CUDA graph; cache copies rotated (ring of 2 to 8 copies, 96 MB target) so every call reads DRAM.

## Cache format and bytes per token per layer

| format | bytes / token / layer |
|---|---|
| compressed, R=256: FP4 latent 128 + E4M3 block scales 16 + FP8 rope key 256 + kr scale 8 + ks 8 | 416 |
| compressed, R=512: FP4 latent 256 + E4M3 block scales 32 + FP8 rope key 256 + kr scale 8 + ks 8 | 560 |
| baseline: fp16 GQA K and V, 4 KV heads x 256 dims | 4096 |

Compression vs baseline: 9.85x (R=256), 7.31x (R=512).

## Correctness (kernel vs fp32 reference on the dequantized cache)

Gate: max relative error of o_lat < 1e-3 (relative = max abs error / max abs reference).

| R | T | S | rel error (kernel vs dequant ref) | status |
|---|---|---|---|---|
| 256 | 1 | 1000 | 8.0e-07 | PASS |
| 256 | 1 | 4096 | 4.7e-07 | PASS |
| 256 | 1 | 32768 | 1.1e-06 | PASS |
| 256 | 4 | 1000 | 7.9e-07 | PASS |
| 256 | 4 | 4096 | 2.3e-06 | PASS |
| 256 | 4 | 32768 | 1.5e-06 | PASS |
| 512 | 1 | 1000 | 5.4e-07 | PASS |
| 512 | 1 | 4096 | 1.4e-06 | PASS |
| 512 | 1 | 32768 | 1.0e-06 | PASS |
| 512 | 4 | 1000 | 1.5e-06 | PASS |
| 512 | 4 | 4096 | 1.4e-06 | PASS |
| 512 | 4 | 32768 | 2.0e-06 | PASS |

For information only, error of the quantized cache against the unquantized fp32 cache is 0.12 to 0.30 relative. This is the FP4 quantization error on Gaussian test data; it is not a kernel error.

## Speed (per call, GPU time)

`GB/s` = cache bytes read per call / time. `% copy` = GB/s / 241.8.

T=1:

| S | SDPA fp16 GQA us | SDPA GB/s (% copy) | kv4 R=256 us | GB/s (% copy) | speedup | kv4 R=512 us | GB/s (% copy) | speedup |
|---|---|---|---|---|---|---|---|---|
| 4096 | 434.4 | 38.6 (16.0%) | 61.2 | 27.8 (11.5%) | 7.1x | 100.2 | 22.9 (9.5%) | 4.3x |
| 32768 | 3487.6 | 38.5 (15.9%) | 372.2 | 36.6 (15.1%) | 9.4x | 652.9 | 28.1 (11.6%) | 5.3x |
| 131072 | 13772.9 | 39.0 (16.1%) | 1462.2 | 37.3 (15.4%) | 9.4x | 2589.4 | 28.3 (11.7%) | 5.3x |
| 524288 | 55009.6 | 39.0 (16.1%) | 6096.8 | 35.8 (14.8%) | 9.0x | 10403.1 | 28.2 (11.7%) | 5.3x |

T=4 (independent sequences, same context length; 524288 not run at T=4, memory):

| S | SDPA fp16 GQA us | SDPA GB/s (% copy) | kv4 R=256 us | GB/s (% copy) | speedup | kv4 R=512 us | GB/s (% copy) | speedup |
|---|---|---|---|---|---|---|---|---|
| 4096 | 541.3 | 124.0 (51.3%) | 202.1 | 33.7 (13.9%) | 2.7x | 360.1 | 25.5 (10.5%) | 1.5x |
| 32768 | 3800.8 | 141.3 (58.4%) | 1507.2 | 36.2 (15.0%) | 2.5x | 2673.2 | 27.5 (11.4%) | 1.4x |
| 131072 | 15144.6 | 141.8 (58.6%) | 6263.9 | 34.8 (14.4%) | 2.4x | 10545.2 | 27.8 (11.5%) | 1.4x |

## Reading the numbers

- The compressed kernel reads 7.3x to 9.9x fewer bytes than the fp16 GQA baseline, and it is 1.4x to 9.4x faster in wall time, depending on T and R. The time win is smaller than the byte win at T=4 and R=512.
- The kernel reaches only about 11% to 15% of measured copy bandwidth. The SDPA baseline reaches 16% at T=1 and 51% to 59% at T=4. The SDPA baseline uses PyTorch's memory-efficient kernel with the 16 query heads laid out as 4 rows per KV head; it was not tuned, so its T=1 number is low.
- Hypothesis, not measured: the kernel is compute-bound on CUDA-core FP32 FMA. Per token it does 16 x R multiply-adds for the scores and 16 x R for the output, plus dequantization. At R=512 this is roughly twice the DRAM time on a T4. The lower % at R=512 fits this. A tensor-core path (int8 or fp16 with split precision) would be the next lever. Not profiled.
- Speedup over SDPA at T=4 is smaller because SDPA's batched kernel is more efficient there.

## Notes and limits

- Correctness is checked on the dequantized cache, so the numbers isolate kernel arithmetic from quantization error.
- The first Kaggle compile was the first real nvcc build of kv4_attn.cu; it compiled and ran without changes.
- Not done: profiling (ncu) to confirm the bound; a tensor-core variant; tuning of TT, the block size and the split target; T > 4.

## v2 (tensor-core, kv4_attn_tc) — measured, results/attn_bench_v2 (verified by Opus: all_correct True)
| S | R | T=1 v1 µs | T=1 v2 µs | v2 % copy BW | v2 vs SDPA fp16 GQA |
|---|---|---|---|---|---|
| 32768 | 256 | 377 | 221 | 25.5 % | 15.7x |
| 131072 | 256 | 1461 | 737 | 30.6 % | 18.7x |
| 524288 | 256 | 6082 | 2651 | 34.0 % | 20.7x |
| 131072 | 512 | 2605 | 1258 | 24.1 % | 10.9x |
| 524288 | 512 | 10527 | 4544 | 26.7 % | 12.1x |
Still below the bandwidth limit: next suspects are the FP4→fp16 smem dequant and per-sequence processing at T=4 (T=4 ≈ 4x T=1).
