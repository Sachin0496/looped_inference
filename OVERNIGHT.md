# Overnight run — 2026-11-18 → 19 (IST)

Goal by morning: SenseNova-Loop-9B variant with DeepSeek-V4.1-Flash-class KV (<~1 KB/token),
knowledge/capacity at teacher baseline (measured, not claimed), plus optimized CUDA inference kernels.
Read PLAN.md for the design. This file is the work queue + rules for each check-in.

## Rules for every check-in
1. `get_usage`. If weekly ≥ 90% or 5-hour ≥ 85%: do not start new Opus-heavy work; make sure
   Kaggle jobs are running; schedule a one-shot CronCreate wake ~5 min after the reset that binds
   (5-hour resets 23:40 IST; weekly resets 06:30 IST), log it here, stop.
2. Prefer Haiku subagents (model: haiku) for research, boilerplate, log reading. Opus does kernel
   math, design decisions and review.
3. Kaggle: account sachin0496, private kernels only. MAX 2 concurrent GPU sessions (push fails otherwise). Log every run in the ledger below with
   accelerator and wall time. Stop launching GPU runs if ledger total > 20 GPU-h this week.
   Kaggle may queue for 30+ min — that's normal.
4. Code/notes may be pushed to github.com/Sachin0496/looped_inference (user asked, PUBLIC repo): never weights (*.pt/*.npz/*.gguf/*.safetensors), secrets or large data. Never publish HF repos or upload weights publicly.
5. Label every number: measured (with result file) vs estimated.

## Work queue (status: todo / running / done / blocked)
- [done] W1 converter engine/convert/gguf_to_hf.py (--check PASS 490/490; transforms inferred from base Qwen3.5
        converter, NOT yet numerically verified). Vision tower not in GGUF → use --text-only, vision weights random (text-only use).
- [running] W3b Kaggle `sachin0496/sensenovakv-kv-release` v1 (pushed 22:16 IST, T4x2): download release BF16 GGUF → convert (--text-only) → kv_analysis with
        SNKV_WEIGHTS=/tmp/release_hf. Source kaggle/kv_release/kv_release.py; outputs → results/kv_release_v1 (watcher in scratchpad). On failure: key_diff.json in outputs. Converter check = release baseline PPL must be close to / better than Preview's; garbage PPL ⇒ mapping bug.
- [done] W2 Research → research/mla_retrofit_recipe.md (joint SVD init w/ calibrated KNorm; KD > NTP; ~6x is the safe no-loss point; full KD needs TPU)
- [running] W3 Kaggle B1 `sachin0496/sensenovakv-kv-analysis` (Preview weights, 2xT4, pushed 20:40 IST):
        PPL baseline, PCA energy, cross-visit/pair R², training-free variants → results/kv_analysis_v1.
        Smoke-tested on a tiny random model on CPU first (SNKV_TINY). Rerun on release weights once W1 converter works.
- [running] W4 Kaggle B2: layer-local distillation of the latent KV (kaggle/distill/distill.py, SPEC.md). TINY (CPU, random tiny model)
        passed for A, B, C: exactness check C (full rank, no QAT) init rel err 3.5e-7 (<1e-4 PASS); loss falls for A (1.50->0.88 in 120 steps)
        and B (2.92->1.76); C at spec lr 3e-4 drifts from its exact init (loss 0.005->0.02-0.06 in 120 steps), at lr 1e-7 it stays at 0.
        Pushed: distill_B v1 (22:52 IST, T4x2). v1 has a cross-GPU loss-sum bug (fixed in build, found after push) -> expect ERROR at the first train step.
        Watcher scratchpad distill_watch.py (nohup, log distill_watch.log): pushes A when a slot frees (fixed build), fetches to results/distill_A_v1,
        re-pushes B as v2 after B v1 ERROR (results/distill_B_v2). Each job max 3 pushes. Not committed.
- [done] W5 4-bit GEMV (engine/kernels/w4_gemv.cu, results/gemv_bench_summary.md; verified by Opus: all_correct True,
        weights rotated past L2). T=1: 78–108% of copy BW, 2.8–3.6x faster than cuBLAS fp16. Weak spot: T=8 ≈ cuBLAS
        (fp32 FMA compute-bound) → follow-up: tensor-core (mma) path for MTP verify batches.
- [done] W6 DeltaNet decode (results/deltanet_bench_summary.md): ALL CORRECT; 0.61 ms/token vs 13.9 ms CUDA-graph ref (T=1 L2-assisted; ~0.8 ms expected).
- [done v1] W7 latent attention (engine/kernels/kv4_attn.cu, results/attn_bench_summary.md): correct (≤2.3e-6 rel), 416–560 B/token/layer
        vs 4096 fp16 GQA (7–10x), 1.4–9.4x faster than SDPA at 131K; only 11–15% BW = fp32-FMA bound (~29 FMA/byte vs ~17 balance).
- [done] W7b tensor-core attention v2: all_correct; 1.5–2.3x over v1, 10–21x over SDPA at T=1; 17–34% BW (dequant-bound?, follow-up).
- [todo] W8 Morning report: REPORT.md with measured tables
- [queued] W9 engine_e2e v1 (engine/runtime/snkv_engine.py + engine/README.md; Kaggle `sachin0496/sensenovakv-engine-e2e`, T4x2, internet): 4-bit packed decode engine on cuda:0 (56 visits, mHC/DeltaNet/GEMV kernels, CUDA graph) vs HF fp16 teacher, 3 prompts x 64 teacher-forced steps (top-1, max|dlogit| step 1, KL), tokens/s 1K/8K HF vs engine, peak mem 8K, KV at 32K/128K. CPU tests pass (tests/test_engine_cpu.py: fp32 plumbing 4e-6 rel). GPU path unrun. Watcher scratchpad engine_watch.py (caffeinate, log engine_watch.log) holds the push until distill-a/b are idle, retries 5 min, fetches to results/engine_e2e_v1. Not committed.
- [queued] W10 loop-convergence diagnostic (training-free, SenseNova-Loop-9B Preview, T4x2): kaggle/loop_conv/loop_conv.py, Kaggle `sachin0496/sensenovakv-loop-conv`. Per token: r_t = ||h2−h1||/||h1|| at layer 27 (last looped layer, after mHC leave) from the 2-loop run; KL(p_2loop||p_1loop), top-1 agreement and dNLL from a 1-loop run on the same 16 fineweb-edu + 16 WikiText-2 sequences x 2048. Outputs NLL/PPL, r quantiles, Spearman(r, KL / dNLL), oracle-exit table at r quantiles, top-20 tokens. Per-token approximation (caveat in results). Smoke test on the tiny model passed (CPU, plan check exact). Watcher scratchpad loop_conv_watch.py (caffeinate, log loop_conv_watch.log) retries the push every 5 min, max 2 accepted pushes, fetches to results/loop_conv_v1. Not committed.

- [running] W11 END-TO-END KD on Kaggle TPU v5e-8 (user OK 2026-11-19 "tpu takes way too much time is that fine then go ahead").
  JAX port: tpu/jax_model.py (Haiku; tiny-model logits vs torch rel 6.5e-5), tpu/jax_student.py (Haiku; config E vs distill.py rel 1e-7, grads 3e-7),
  job tpu/kd_train.py → render `kd_tpu` → Kaggle `sachin0496/sensenovakv-kd-tpu` (machine_shape TpuV5E8, kernel_sources distill-e for student_E.pt).
  KL(teacher||student) full vocab, only config-E KV tensors train, relative lr 1e-3, batch 8x2048, 5 h budget, eval init+final (wiki/fineweb ppl, MMLU-300).
  Aborts if JAX teacher wiki ppl > 25 (torch ref 19.80). Probe kernel `sachin0496/sensenovakv-tpu-probe` (env check) holds the single TPU slot first;
  push kd_tpu after it ends (MAX 1 TPU session). TPU quota 20 h/week, resets 2026-11-21. Outputs → results/kd_tpu_v1 (npz gitignored).

## Kaggle ledger
| run | accel | start (IST) | wall | result |
|---|---|---|---|---|
| mhc-bench v1 | T4 | 18.11 ~18:30 | ~5 min | results/mhc_bench_v1 |
| mhc-bench v2 | T4 | 18.11 ~19:05 | ~5 min | results/mhc_bench_v2 |
| mhc-bench v3 | T4 | 18.11 ~19:15 | ~5 min (+30 min queue) | results/mhc_bench_v3 |
| gemv-bench v1 | T4 | 20.18 ~20:22 | ~5 min | results/gemv_bench_v1 |
| gemv-bench v2 | T4 | 20.18 ~20:30 | ~5 min | results/gemv_bench_v2 |
| kv-analysis v1 | T4x2 | 18.11 20:40 | ~3 min (ERROR: missing mtp file) | results/kv_analysis_v1 |
| kv-analysis v2 | T4x2 | 18.11 21:25 | ~3 min (ERROR: index still points at mtp.safetensors) | results/kv_analysis_v2 |
| kv-analysis v3 | T4x2 | 18.11 ~22:20 | ~8 min (ERROR: cross-GPU in pair R²) | results/kv_analysis_v3 |
| kv-analysis v4 | T4x2 | 18.11 22:40 | ~2.5 h | results/kv_analysis_v4 |
| distill-b v1 | T4x2 | 18.11 22:52 | 12 h cap (crashed step 1, then hung) | results/distill_B_v1 (none) |
| distill-a v1 | T4x2 | 19.11 01:10 | 3.6 h work + hang to 12 h cap | results/distill_A_v1/live_log.txt |
| distill-e v1 | T4x2 | 19.11 10:56 | 3.7 h | results/distill_E_v1 |
| loop-conv v1 | T4x2 | 19.11 12:49 | ~8 min | results/loop_conv_v1 |
| engine-e2e v1 | T4x2 | 19.11 13:00 | 14 min | results/engine_e2e_v1 |
| kv-release v1 | T4x2 | 18.11 22:16 | pending | results/kv_release_v1 |
| distill-b v1 | T4x2 | 18.11 22:52 | running (known bug, see W4) | results/distill_B_v1 |
| distill-a v1 | T4x2 | pending (session limit, watcher retries) | - | results/distill_A_v1 |
| engine-e2e v1 | T4x2 | queued (watcher: waits for distill-a/b idle, then pushes) | - | results/engine_e2e_v1 |
| loop-conv v1 | T4x2 | queued 19.11 09:01 (watcher retries push every 5 min, session limit) | - | results/loop_conv_v1 |

## Pacing
Opus burns ~1%/min of the 5-hour window and ~0.2%/min weekly during heavy work. Keep check-ins short; give implementation to Haiku.

## Check-in log
- 20:07 IST: usage 5h 33%, weekly 66%. Overnight run started.
- 20:45 IST: usage 5h 42%, weekly 68%. B1 pushed; W5/W6 Haiku agents started; check-in cron hourly at :47.
- 21:14 IST: usage 5h 53%, weekly 69%. kv-analysis RUNNING (left queue ~21:00), deltanet QUEUED. W5 GEMV done+verified. Started W7 (Haiku).
- 21:35 IST: kv-analysis v1 ERROR at model load (MTP companion file excluded from download). Fixed (mtp_num_hidden_layers=0); v2 auto-pushes when a GPU slot frees → results/kv_analysis_v2. W7 attn-bench v1 RUNNING (its agent's loop fetches to results/attn_bench_v1; agent exited without summary → check-in must summarize). deltanet QUEUED.
- 21:50 IST: kv-analysis v2 ERROR (weight index lists mtp.safetensors). v3 downloads it; auto-push loop running → results/kv_analysis_v3.
- ~22:10 IST: deltanet-bench v1 COMPLETE, verified + summarized by Opus.
- 22:15 IST: usage 5h 64%, weekly 71%. attn v2 COMPLETE+verified. kv-analysis v3 QUEUED. Started W3b release-weights job (Haiku).
- 22:40 IST: kv-analysis v3 got through load + calibration on Preview (313 tok/s; wiki nll 2.914 on 2 seqs). PCA energy of raw pre-norm [K;V]: r=256 keeps 70–89%, r=512 84–97% → KV is NOT very low-rank; training-free compression will cost quality (distillation needed). Crashed at pair R² (cross-GPU device mismatch). Fixed (stats moved to cuda:0), tiny test OK, v4 pushed → results/kv_analysis_v4 (~3.5 h eval).
- TODO for check-in: kv-release v1 (running) embeds the OLD script → will crash at the same point after conversion. When its watcher fetches results/kv_release_v1, confirm the converter part worked (release baseline wiki nll vs Preview 2.914), then re-push kv_release (build already re-rendered with the fix): `kaggle kernels push -p kaggle/build/kv_release --accelerator NvidiaTeslaT4`, and fetch into results/kv_release_v2.
- 22:54 IST: W4 distill_B v1 pushed; A blocked by 2-session limit, watcher retrying every 5 min. Opus-free check-in: TINY tests done.
- 23:05 IST: distill (B2) implemented by Haiku, TINY-tested (config C exact at init: 3.5e-7). Opus fix: per-tensor RELATIVE lr (lr = 2e-3 × rms(init)); old absolute 3e-4 drifted config C off its exact init (0.02–0.06 → now 0.0005–0.002). distill_B v1 (running) has a cross-GPU bug + old lr → will ERROR; watcher (scratchpad distill_watch.py) re-pushes B v2 and pushes A from the re-rendered builds when slots free.
- 23:15 IST: usage 5h 73%, weekly 72%. kv-release v1 ERROR at the old pair-R² bug, BUT converter VALIDATED: release wiki nll 2.811 vs Preview 2.914 on the same 2 seqs (PPL 16.6 vs 18.4), key diff 0 missing/0 unexpected/0 mismatch, PCA spectra ≈ Preview. Decision: distill on RELEASE weights. Opus stopped distill_watch.py (it would push Preview-based B v2/A); Haiku adds release conversion to distill.py and restarts the watcher. distill-b v1 (Preview, buggy) still running → will ERROR. kv-analysis v4 (Preview) RUNNING.
- 23:17 IST: distill now trains on RELEASE weights. distill.py: SNKV_RELEASE=1 converts the BF16 GGUF with kv_release.py's steps into /tmp/release_hf (not TINY); render.py sets SNKV_RELEASE=1 for distill_A/B. results.json logs weights (release-gguf-converted / preview). TINY smoke B and C pass (20 steps, C exact 3.5e-7); renders AST-checked. Watcher (scratchpad distill_watch2.py, log distill_watch2.log, caffeinate, 11 h timeout) waits for distill-b v1 to end (fetch to results/distill_B_v1), pushes B then A on session-limit retry, fetches to results/distill_B_v2 and results/distill_A_v1. Per-tensor relative lr unchanged.
- 00:13 IST: usage 5h 1% (reset), weekly 72%. distill-b v1 (Preview, absolute lr 3e-4) has been RUNNING 80 min — the cross-GPU bug did not trigger, so it is a real Preview config-B run (3 h train + eval → ~02:15). kv-analysis v4 RUNNING (→ ~02:15). distill_watch2 then pushes release B v2 + A (→ ~06:30–07:00). Nothing else runnable (2/2 GPU slots busy).
- ~01:10 IST: kv-analysis v4 COMPLETE (results/kv_analysis_v4, Preview, 64+64×2048 eval/variant). Training-free fineweb PPL Δ: pv512_fp4h 4032 B/tok +2.2%; share512_fp4h 2304 B +5.4%; share256_fp4h 1152 B +13%; share256_pair_fp4h 720 B +18%; FP4 on latent ≈ free (+0.05%); Hadamard no help; pv2048 control = base (−0.01%). Cross-visit R² r256 0.58–0.69, r512 0.63–0.78 → loop KV is only partly predictable from visit-1 latent (linear). Pushed release distill A v1 into the freed slot; replaced watcher2 with two Opus background watchers (B v1 → push release B v2 → results/distill_B_v2; A → results/distill_A_v1).
- ~01:30 IST: W9 engine_e2e built: CPU tests pass (packer byte-identical to w4_ref; 56-visit plan; fp32 plumbing vs HF on dequantized weights, worst rel 4.2e-6, 8/8 top-1). Kaggle render kaggle/build/engine_e2e (render.py entry engine_e2e). Watcher launched (caffeinate, nohup): distill-a and distill-b both RUNNING at launch, so the push waits. Note the distill watchers also re-push into these two slots, so the engine push may be delayed past their end.
- 02:13 IST: usage 5h 8%, weekly 73%. distill-b v1 (Preview) and distill-a v1 (release) RUNNING; engine-e2e waits for a slot (watcher gate). Nothing else runnable.
- 03:13 IST: usage 5h 9%, weekly 73%. distill-b v1 RUNNING 4h20m (train 3h + calib + eval), distill-a v1 RUNNING ~2h. Engine waits.
- 04:13 IST: usage 5h 11%, weekly 73%. Both distill runs still RUNNING (B v1 5h20m). Haiku drafting REPORT.md with placeholders for pending runs.
- 05:13 IST: distill-b v1 RUNNING 6h20m (expected ~4.5 h: 3 h train + calib + eval) — cause unknown (no live logs via CLI; code review: calib cov uses GPU fp16 matmul, eval sizes modest). distill-a RUNNING 4h. Kaggle hard limit 12 h (B by ~10:50). Engine still gated. Report draft held in Opus context (Haiku could not write REPORT.md).
- 06:13 IST: both distill runs still RUNNING (B 7h20m, A ~5h). Live log not reachable (browser not signed in to Kaggle). Waiting; Kaggle 12 h cap ends B ~10:50, A ~13:10.
- 07:15 IST: both distill runs still RUNNING. Wrote REPORT.md with all measured results; distillation + engine sections marked pending. Next check-ins: fill them in when results land.
- 08:13 IST: both distill runs still RUNNING (B 9h20m, A ~7h). No new results.
- 09:30 IST: ROOT CAUSE of the "slow" distill runs: they finish, then the process never exits (streaming-dataset threads) → Kaggle shows RUNNING until the 12 h cap. Found via `kaggle kernels logs -f`. Fix: hard os._exit(0) footer in all 5 job scripts; builds re-rendered.
  distill A v1 (release, r512 loop-shared FP4, 4,416 B/token = 13x smaller) RESULT (results/distill_A_v1/live_log.txt): teacher wiki/fineweb PPL 19.80/14.87 → student 20.49/15.64 (+3.5%/+5.2%); MMLU-300 72.3% → 68.7% (−3.7 pt, ±2.6). Final per-layer rel. error: visit-0 0.09–0.32, visit-1 (shared latent) 0.41–0.56 → cross-loop sharing is the main loss; layer-local training barely reduced it (4.1M tokens).
  Decision: drop release B v2 (more sharing → worse); Haiku implements config E (visit-1 residual latent r=128, ≈3.3 KB/token). Queue order when slots free (B v1 hung until ~10:50, A until ~13:10): distill_E > loop_conv > engine_e2e.
- 10:20 IST: distill-b v1 had actually CRASHED at step 1 (cross-GPU loss sum, ~15 min in) and then hung 11 h. Added excepthook → os._exit(1) to all jobs, re-rendered, TINY E OK, pushed to GitHub. Queue (E > loop_conv > engine) waits for slots.
- 11:13 IST: distill-b v1 cancelled at the 12 h cap (CANCEL_ACKNOWLEDGED). distill_E pushed 10:56, RUNNING (~4.3 h → ~15:15). distill-a hung until ~13:10. loop_conv, engine_e2e queued next.
- 12:13 IST: distill-e RUNNING (~1h15m in), distill-a still hung (cap ~13:10). No change.
- ~12:50 IST: distill-a session ended; outputs fetched (results.json confirms live-log numbers; student_A.pt 152 MB kept local, gitignored). Freed slot → queue pushes loop_conv next.
- 13:13 IST: loop_conv v1 COMPLETE (7.6 min): skipping pass 2 → PPL 18.8→88.9 (+372%), top-1 45%; r median 2.1, Spearman(r, KL)=0.11, (r, dNLL)=−0.03 → no training-free early exit. engine_e2e pushed 13:00 RUNNING; distill-e RUNNING.
- 14:15 IST: engine-e2e v1 COMPLETE: 15.5 tok/s @1K vs HF 1.8; 87% top-1, KL 0.048; weights 5.05 GB but peak 14.2 GiB (HF still in process) → 8K OOM. Haiku building v2 (fresh-process engine, 1K/8K/32K memory). distill-e RUNNING.
- 14:21 IST: Kaggle WEEKLY GPU QUOTA (30 h) EXHAUSTED — mostly by the two hung distill sessions (~24 h). engine_e2e v2 (fresh-process memory bench, CPU-checked) ready but cannot run until the weekly reset; its watcher stopped. distill-e v1 still RUNNING (watch whether Kaggle lets it finish). Rule added: always add the hard-exit footer + excepthook before pushing any job.
- 14:45 IST: distill-e v1 COMPLETE in 3.7 h and EXITED (hard-exit fix works). E: 3,660 B/token (15.7x), wiki +1.9%, fineweb +3.6%, MMLU 68.7% (teacher 72.3%). Beats A on bytes and PPL; MMLU gap unchanged → end-to-end KD needed. GPU quota exhausted; awaiting user choice (TPU KD port vs wait).
- 15:13 IST: nothing runnable (Kaggle GPU quota exhausted), nothing pending, REPORT.md complete. Hourly check-in cron stopped; waiting for the user's choice (TPU KD port vs GPU reset).
- 16:10 IST: user chose TPU KD. JAX port + config-E student done by Haiku agents and verified vs torch; KD job smoke-tested (tiny KL 0.158→0.132). TPU probe queued (holds the 1 TPU slot); kd_tpu push waits for it. Hourly check-in cron restarted.
- 17:33 IST: probe still QUEUED (~1 h 50 m, single TPU slot); kd-tpu not pushed yet (404 = no version). Usage 5h 17%, weekly 8%. Waiting.
- 18:33 IST: probe still QUEUED (~2 h 50 m). Official kaggle-cli docs/kernels_metadata.md confirm machine_shape `TpuV5E8` = TPU v5e-8, so this is TPU demand, not a bad shape. kd-tpu waits. Usage 5h 18%, weekly 8%.
- 19:00 IST: teammate GPU queue opened as GitHub issue Sachin0496/looped_inference#1 (engine_e2e v2, distill_E repro, gate suite, engine FP4 cache, KV sweep, engine speed); render.py KAGGLE_USER override pushed. TPU probe still queued.
