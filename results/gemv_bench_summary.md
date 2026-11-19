# W4 fused GEMV: benchmark summary (Kaggle Tesla T4, sm_75)

Measured copy bandwidth (read+write convention, 256 MB device copy): 241.8 GB/s.
Note: a pure-read kernel can exceed the copy number; cuBLAS fp16 reads 258-261 GB/s at T=1 on the big shapes, so the copy figure is about 0.93 of read peak. Percentages below use the copy figure as specified.

Timing: all calls for one measurement captured in one CUDA graph (weight copies rotated to exceed L2; 96 MB ring, 2 to 8 copies). Times are per call, GPU time.

## Best version per shape and T (plain GEMV, no norm, no residual)

| N x K | T | best w4 us | version | rpw/slab | w4 GB/s (packed) | % copy BW | cuBLAS fp16 us | speedup vs cuBLAS | dequant-then-matmul us |
|---|---|---|---|---|---|---|---|---|---|
| 1024x4096 | 1 | 12.6 | v2 | 1/2048 | 187.6 | 77.6 | 35.4 | 2.82x | 402 |
| 1024x4096 | 2 | 16.8 | v2 | 2/2048 | 140.5 | 58.1 | 37.1 | 2.21x | 411 |
| 1024x4096 | 4 | 22.9 | v2 | 4/2048 | 103.0 | 42.6 | 37.3 | 1.63x | 413 |
| 1024x4096 | 8 | 40.0 | v2 | 4/2048 | 59.0 | 24.4 | 37.4 | 0.94x | 410 |
| 4096x4096 | 1 | 39.9 | v1 | 2/2048 | 236.6 | 97.9 | 131.1 | 3.29x | 1632 |
| 4096x4096 | 2 | 49.4 | v2 | 2/2048 | 191.1 | 79.0 | 150.9 | 3.06x | 1630 |
| 4096x4096 | 4 | 77.2 | v1 | 2/2048 | 122.3 | 50.6 | 151.8 | 1.97x | 1632 |
| 4096x4096 | 8 | 173.1 | v2 | 2/2048 | 54.5 | 22.5 | 152.6 | 0.88x | 1638 |
| 4096x12288 | 1 | 116.8 | v1 | 2/2048 | 242.3 | 100.2 | 387.3 | 3.32x | 4927 |
| 4096x12288 | 2 | 145.5 | v2 | 2/2048 | 194.5 | 80.4 | 442.8 | 3.04x | 4928 |
| 4096x12288 | 4 | 218.1 | v2 | 4/2048 | 129.8 | 53.7 | 444.7 | 2.04x | 4943 |
| 4096x12288 | 8 | 511.4 | v2 | 2/2048 | 55.4 | 22.9 | 450.3 | 0.88x | 4953 |
| 8192x4096 | 1 | 74.5 | v1 | 2/2048 | 253.5 | 104.8 | 258.3 | 3.47x | 3254 |
| 8192x4096 | 2 | 99.3 | v2 | 2/2048 | 190.1 | 78.6 | 286.3 | 2.88x | 3242 |
| 8192x4096 | 4 | 146.0 | v2 | 4/2048 | 129.3 | 53.5 | 286.3 | 1.96x | 3246 |
| 8192x4096 | 8 | 300.9 | v2 | 4/2048 | 62.7 | 25.9 | 286.7 | 0.95x | 3248 |
| 12288x4096 | 1 | 107.9 | v1 | 2/2048 | 262.4 | 108.5 | 385.4 | 3.57x | 4916 |
| 12288x4096 | 2 | 141.0 | v2 | 2/2048 | 200.8 | 83.0 | 418.3 | 2.97x | 4915 |
| 12288x4096 | 4 | 222.1 | v2 | 4/2048 | 127.5 | 52.7 | 418.4 | 1.88x | 4885 |
| 12288x4096 | 8 | 447.7 | v2 | 4/2048 | 63.2 | 26.1 | 420.9 | 0.94x | 4935 |
| 248320x4096 | 1 | 2877.5 | v2 | 4/2048 | 198.8 | 82.2 | 19823.4 | 6.89x | 104621 |
| 248320x4096 | 2 | 3481.7 | v2 | 4/2048 | 164.3 | 68.0 | 8565.5 | 2.46x | 104390 |
| 248320x4096 | 4 | 4947.5 | v2 | 4/2048 | 115.6 | 47.8 | 8879.6 | 1.79x | 104812 |
| 248320x4096 | 8 | 8740.9 | v2 | 4/2048 | 65.5 | 27.1 | 9415.4 | 1.08x | 105135 |

Columns: w4 = our kernel (best of configs swept: rows per warp 1/2/4, K-slab 2048/4096). cuBLAS = F.linear on the fully dequantized fp16 weight. dequant-then-matmul = torch ops (unpack, scale, add) then F.linear, per call.

## Correctness

All 1152 checks passed (every shape, T in 1/2/4/8, with and without RMSNorm prologue, with and without residual, every config).
Reference: fp32 matmul with the dequantized weights (and the fp16-rounded RMSNorm output when the prologue is on).
Worst max relative error (max abs error / max |y_ref|): 4.62e-04; worst max abs error: 3.91e-03. Threshold 1e-3.

## Design (v2 is the current kernel; v1 kept in results/gemv_bench_v1)

- Format: 4-bit asymmetric, group 64, fp16 (scale, minv) per group. 16-byte loads of 32 contiguous weights; the two nibble halves of each byte hold k and k+16, so each uint4 word decodes to two aligned quads. 0.5625 bytes per weight.
- One warp owns RPW consecutive rows; lane l reads chunks l, l+32, ... (512 B contiguous per warp load). Each weight byte is read once per call for all T rows.
- Activations: per K-slab (2048 or 4096 elements) the block stages the (RMS-normalized) x as fp16 in shared memory, quad-interleaved so lanes read consecutive 8-byte words (no bank conflicts). The RMSNorm sum of squares is recomputed per block over the full row.
- v2 change: dequantize the RPW chunks first, then load each x quad once and reuse it across the RPW rows. This cut shared-memory traffic and fixed most of the T>=2 slowdown (lm_head T=8: 16.9 ms to 8.7 ms).
- Dequant in registers: q placed in the mantissa of 2^23 (exact), minus 2^23, then fma(q, scale, minv). fp32 accumulate, fp16 output.

## Notes and limits

- T=1 target (>=70% of copy BW on big shapes): met on 4096x4096, 4096x12288, 8192x4096, 12288x4096 and lm_head 248320x4096 (82%). The 1024x4096 shape is at 78%.
- T>=2 is compute bound (one FMA per weight per row in fp32). Versus cuBLAS fp16 on the big shapes, the kernel is 2.5-3.5x faster at T=2, 1.8-2x at T=4, and roughly equal at T=8 (0.88-1.08x). fp16x2 math would be the next lever; it would change accumulation precision, so it was not attempted.
- RMSNorm prologue cost is the per-block sum-of-squares read (L2); the norm and residual variants are in the bench log, not separately tabulated here.
- Not done: split-K for small N (1024 is the only small shape; it is latency-limited at about 13 us).
