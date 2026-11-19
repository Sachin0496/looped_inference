# Fused Gated DeltaNet decode step — T4 results (deltanet_bench_v1, measured)

Kernel: engine/kernels/deltanet_fused.cu (conv update + SiLU, l2norm, g/beta, gated delta rule, gated RMSNorm; one launch).
Reference: engine/ref/deltanet_ref.py (transformers qwen3_5 decode path).

Correctness (64 carried decode steps): ALL CORRECT. Output max abs error 7.0e-4 of max|out| (bf16 ulp there = 0.17),
≥99.99% of elements within 1 bf16 ulp; final fp32 recurrent state max error 1.5e-7 relative; conv state exact.

| T (sequences) | ref eager µs | ref CUDA graph µs | fused µs | per-seq-step µs | state BW / copy BW | speedup vs ref graph |
|---|---|---|---|---|---|---|
| 1 | 1084 | 331 | 14.6 | 14.6 | 118 %* | 22.7x |
| 4 | 1033 | 650 | 82.9 | 20.7 | 83 % | 7.8x |
| 16 | 2185 | 2178 | 312 | 19.5 | 89 % | 7.0x |

Per decoded token (42 DeltaNet layer visits): eager 45.5 ms, CUDA-graph reference 13.9 ms, fused kernel 0.61 ms.

*Caveat (Opus): at T=1 the 2 MiB fp32 state fits the T4's 4 MiB L2 and the benchmark replays the same layer, so 118 % is
L2-assisted. In real decode 42 distinct states (84 MiB) stream from DRAM; T=4/16 (state > L2) show ~85–89 % → expect
~17–20 µs per visit, ~0.8 ms per token. Re-measure with rotated states before quoting the T=1 number.
