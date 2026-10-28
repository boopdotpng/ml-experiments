# architectures: GPT-2 in 2026

The project plan is below; the code so far is only the data step.

## Files

- `data.py`: downloads and tokenizes the data mix into llm.c-style uint16 shards, plus `TokenLoader`. `python data.py --subset` writes ~50M tokens to `data-subset/` for testing shapes and train steps locally before spending H100 hours; `python data.py` writes the full 20B to `data/`. Both directories are gitignored.
  - The tokenizer is GPT-2 BPE (tiktoken) as a stand-in until the project's own 32,768-vocab BPE exists; plug it into `get_encoder()`. Don't hard-code vocab 50257 anywhere else: read `vocab_size` from `meta.json`.
  - Web is Nemotron-CC **v1** (quality=high, organic only), streamed ungated from `data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/`, because HF access to `nvidia/Nemotron-CC-v2.1` is still pending. Its files are taken in a fixed shuffled order so they span every crawl. `nvidia/Nemotron-CC-Math-v1` is gated on HF; a gated source without access is skipped with a warning.
  - `HuggingFaceTB/stack-edu` only has blob IDs, not file contents, so code comes from `common-pile/stackv2_edu_filtered`, filtered to `int_score >= 3`. That copy keeps only openly licensed repos, so it is a subset of Stack-Edu.

---

# GPT-2 in 2026: Project Plan

Oct 26, 2026

## Goal

Build a fast, correct, hand-written training stack on NVIDIA GPUs and prove it on four real models. Learn the main 2024–2026 architectures by building them, train ~350M-parameter models on 20B tokens at 4k context (16k at the end), and produce one distinctive research result on very long context with linear attention. The ladder is the test bed; the stack, the kernels and the measured numbers are the product.

PyTorch plus custom kernels, not a C++ stack: models, training loop, optimizer, distributed layers and kernels are written by hand; libraries are used only after their code is read and verified. Plumbing that cannot change the results (launch scripts, data download, eval wiring, plots, demo) is automated. RL and detailed interp methods come later.

## Priorities, in order

If something slips, the items at the bottom go first.

1. A fast training stack with measured numbers: single-GPU MFU, then DDP and ZeRO-1 with a scaling-efficiency table, and an analytical performance model that predicts each number before it is measured.
2. The Triton chunked GDN-2 kernel, forward and backward, matched against flash-linear-attention. The main kernel; FLA is the fallback if it is not ready. Plus exactly one raw CUDA kernel as a torch extension.
3. The ladder with its rigor: R2 with 2 seeds, R6, R7, then R4 last; per-architecture LR sweeps, noise floor, bootstrap CIs, scaling curves.
4. One flagship write-up: question 1, very long context with pure linear attention.
5. Cross-hardware numerics: R2 on Blackhole through the tinygrad backend, diffed against H100; FP8 and block-float quality on own weights.

## Research questions

Write a one-line guess before each run.

1. **Very long context with 100% linear attention (flagship).** Runs: R7 (pure GDN-2), R6 (hybrid) for comparison, R2 as the ceiling. Define "reasonable" before the run, e.g. needle recall at 16k within a set margin of R2. Core experiments: run far past training length, recall vs distance, decay-gate half-lives compared with measured recall. Details under Side investigations.
2. **Scaling check: does the ranking hold, and does R7's recall horizon grow with size?** Not a separate study, a rule for reading the ladder. R2, R6 and R7 only (the softmax → hybrid → pure-linear axis; R4 is orthogonal) at ~20M, ~60M and ~350M, the first two being the debug runs with their swept LRs. Two metrics per size: bits per byte, and recall vs distance (needle, MQAR) past training length. A gap at one size means little on its own: it could be closing with size or widening. "Pure GDN-2 trails by 8% at 20M, 4% at 60M, 2% at 350M, and its recall horizon doubles with each size" is a result; "2% behind at 350M" is not. The recall-horizon trend is the part that matters for question 1, since GDN-2's state grows with the model and the question is whether capacity grows fast enough.
3. **Expert prefetch and offload (systems angle, smaller).** Run: R4. Can the next layer's experts be predicted from the current hidden state; how stable is routing across neighboring tokens and layers, i.e. how cacheable it is. What the MLA latent encodes vs R2's GQA keys/values is a secondary interp question on the same checkpoints.

**Search loop, capped at three variants.** Cheap one-flag ablations at debug scale (~20–50M params, ~0.5–1B tokens), each compute-matched against a scaled-down R2 and run alongside the LR sweeps. A variant counts only if it beats the 3-seed noise floor; at most one is promoted to a 20B run. The three: NoPE on R2 (bears on question 1; judge it on recall past training length, not just loss), KDA on R6/R7 (GDN-2 with erase and write gates tied, one flag on the layer and a small kernel change; the Kimi Linear/K3 layer), and QK-norm on R2 (the modern-recipe check that the baseline is not handicapped). Conditional fourth: learned attention sinks, only if R6's sliding-window layers degrade past training length.

What each question is measured by (details under Evaluation): question 1 by distance-conditioned loss on code, loss vs position, and BABILong / RULER tracking and aggregation accuracy, all as a function of distance; question 2 by bits per byte and the question 1 metrics at three sizes; question 3 by next-layer expert prediction accuracy, routing cache hit rate, and the loss cost of serving experts from the cache. A question without a number attached is not finished.

## Fixed settings (identical across the ladder)

Only the architecture changes between ladder runs. Everything below stays constant so differences can be attributed.

| Setting | Value |
|---|---|
| Data mix | Nemotron-CC-v2.1 high-quality ~60% · FinePDFs (English) ~15% · Stack-Edu ~10% · Nemotron-CC-Math ~5% · Wikipedia (Finewiki) ~5% · remainder tunable; long-document sources upweighted in the 16k phase |
| Tokens per run | 20B, same shuffled order and seed |
| Tokenizer | Own byte-level BPE encoder, 32,768 vocab, GPT-4-style pre-tokenization regex. Own trainer time-boxed to a few days; if slow on a few GB, train with the tokenizers library and keep the own encoder |
| Shape | d_model 1,024, 24 layers, 16 query heads, head dim 64, tied embeddings; ~340M dense |
| Context | 4,096 tokens for the first ~90% of tokens, then 16,384 for the final ~10% (inside the WSD decay); RoPE base raised from 10k to 500k at the switch. Documents separated by end-of-text |
| Batch | ~1M tokens per step (micro-batch + gradient accumulation, split across GPUs) |
| Precision | BF16 autocast; norms, softmax, loss, optimizer state, router logits and linear-attention decays in FP32 |
| Optimizer | Muon for hidden 2D matrices, AdamW for embeddings, norms, biases, gates, convs; weight decay 0.1 on matrices only. Muon verified on the debug transformer against a reference implementation |
| Schedule | Warmup-stable-decay: short warmup, constant, final 10–20% decayed to zero |
| Checkpoints | Log-spaced weights-only BF16 copies kept for later interp; latest full state for resuming at the same GPU count |
| Validation | Separate held-out sets per source, a few million tokens each |

The tokenizer is trained once and reused for every run. Comparisons with the original GPT-2 use bits per byte, since the tokenizers differ. The context switch and the decay phase overlap; that is a known confound and is not separated in this project.

**Why this data mix.** Nemotron-CC-HQ outperforms DCLM and FineWeb-Edu in matched-recipe comparisons, with the gap widening at larger sizes. FinePDFs and whole code files are there for length: FineWeb-style web pages average about a thousand tokens, so a 16k phase on web data alone sees mostly concatenated unrelated documents, and question 1 needs real long-range structure in training. Nemotron-CC includes a large synthetic-rephrase fraction; fine for training, worth remembering when comparing with GPT-2.

## Rigor

- **Per-architecture learning rate:** sweep peak LR (and warmup) for each architecture at debug scale, ~4–5 values, and carry the best to 350M. A shared LR would favor whichever architecture it happens to suit.
- **Noise floor:** R2 with 3 seeds at debug scale and 2 seeds at full scale. A difference smaller than the seed spread is not a result.
- **Confidence intervals:** bootstrap CIs on every benchmark number.
- **Correctness:** test each layer against its official implementation (DeepSeek reference code, NVlabs GDN-2, flash-linear-attention), not only against your own naive version.
- **Reproduce one published result:** R6 is the hybrid block design from the GDN-2 paper, so its debug-scale MQAR curve can be checked against the paper before the ladder is trusted.
- **Review the automated parts that can silently bias results:** data filters and eval scoring code get a line-by-line read.

## Building each architecture

Each architecture goes through the same five steps before it gets a GPU.

1. Read the paper and the official code.
2. Derive the math on paper, including the chunkwise form for linear attention.
3. Write a naive, readable PyTorch version: the ground truth and the interp implementation.
4. Test it against the official implementation on random inputs.
5. Swap in a fast kernel only after it matches within numerical tolerance.

Build order: R2 and the distributed stack first, then GDN-2 and its kernel (R6, R7), then R4 last, so a slip costs one run rather than the core result.

## The ladder: 4 core runs

Four 20B-token runs at ~350M. R2 is the control every other run is measured against; the original GPT-2 weights are the reference in evals. The ladder models are the release; there is no separate flagship or continuation run.

| Run | Architecture | Compare with | Measures | Est. GPU-h |
|---|---|---|---|---|
| R2 | Llama 3-style dense: RoPE, RMSNorm, SwiGLU, GQA (4 KV heads), Muon + WSD | Original GPT-2 weights | Control; total gain from data, recipe and architecture | ~25 (×2 seeds) |
| R6 | GDN-2 alternating with 512-token sliding-window GQA (the GDN-2 paper's hybrid block) | R2 | Constant-memory hybrid vs full softmax; reproduces a published design | ~25–30 |
| R7 | Pure GDN-2, no softmax attention anywhere | R6, R2 | Cost of removing softmax; unbounded-length behavior; question 1 | ~25–30 |
| R4 | DeepSeek V3-style: MLA attention + DeepSeekMoE MLP, ~350M active, ~1.5B total | R2 | MLA + MoE at fixed active compute; expert parallel; question 3 | ~40–50 |

GPU-hours assume ~50% MFU on H100 SXM; wall-clock divides by the number of GPUs.

### Configs at this scale

- **R2:** d 1,024, 24 layers, 16 query heads, 4 KV heads, SwiGLU hidden 2,816, no QK-norm (QK-norm is a search-loop variant). Attention via SDPA / FlashAttention; writing your own would not beat it.
- **R6:** GDN-2 with 8 heads of dim 128, short conv size 4, output gate and norm; layers alternate GDN-2 and sliding-window GQA (window 512, RoPE inside the window); memory stays constant at any length. The sliding window is a flag on R2's attention.
- **R7:** R6's GDN-2 layer in all 24 positions; no positional encoding needed.
- **R4:** MLA with KV latent rank 384, decoupled RoPE key dim 32, no-RoPE head dim 64, value head dim 64, no query compression; write the absorbed form too. MLP: first layer dense; other layers 1 shared expert + 64 routed experts, top-6, sized so active MLP width ≈ 2,816. Aux-loss-free bias balancing. Needs the grouped-GEMM kernel and expert parallelism; built last.

### Not in the ladder

V4-style CSA/HCA sparse attention (needs long context to mean anything), Kimi K3's attention residuals, gpt-oss-style window-128 + full attention, standalone MLA with a dense MLP, and R2 with AdamW + cosine are all dropped. NoPE, KDA and QK-norm live in the search loop above.

## Performance

Each run should be fast before it is long, and every number should be predicted before it is measured.

**Analytical performance model.** For each architecture (GQA, MLA, GDN-2, MoE) and each parallelism setup: FLOPs, bytes moved, roofline, predicted step time, memory and MFU. Checked against measurements on every run; every gap is a thing to explain. Written before the first 20B run.

**Single GPU.** Target R2 at ≥ 50% MFU on one H100 SXM: torch.compile, SDPA / flash attention, fused optimizers, batched Newton–Schulz for Muon, memory-mapped uint16 shards, no host syncs in the step loop. Profile each architecture once at debug scale (PyTorch profiler, Nsight Systems/Compute) and fix the top bottleneck before its 20B run. MFU and tokens/s logged next to the loss on every run.

**Kernels.** Each gets an Nsight Compute roofline and a stopping rule (e.g. ≥ 80% of the reference, then move on).

- Chunked GDN-2, forward and backward, in Triton. No PyTorch op exists and the naive recurrence is unusably slow. Matched against flash-linear-attention; FLA is the fallback.
- MoE token permute/unpermute and grouped GEMM, in raw CUDA C++ as a torch extension. The one hand-written CUDA kernel; benchmarked against a Triton version of the same thing.
- Everything else (Muon Newton–Schulz, norms, SwiGLU, loss) stays on torch.compile; profile before touching.

**Distributed**, written by hand on torch.distributed (NCCL). The work is in scheduling and overlap, not transport.

1. DDP across 2–4 H100s on one node (NVLink): gradient bucketing overlapped with backward; scaling-efficiency table, target > 90% at 4 GPUs.
2. ZeRO-1 sharded optimizer state.
3. Expert parallelism for R4: experts spread across GPUs with all-to-all dispatch; throughput compared with plain data parallel. R4's ~1.5B parameters with optimizer state fit on one GPU, so full parameter sharding is not needed.
4. Plain checkpoint resume at the same GPU count is enough for spot instances.

**Cross-hardware numerics.** Run R2 on Blackhole through the tinygrad backend (blackhole-py already runs Llama 3-shaped decode). Diff activations layer by layer against H100, root-cause divergences to specific ops, then measure FP8 and block-float quality on own trained weights. Which architectures survive int8/int4/FP8 best.

## Side investigations

1. **Very long context with R7 (the flagship write-up).** The state is fixed-size, so R7 can run forever with constant memory, but it is lossy memory, not infinite recall.
   - Run R7 and R6 past training length: 16k → 64k → 256k → 1M tokens; check perplexity stays flat and state norms stay bounded.
   - Recall vs distance: needle and MQAR accuracy as a function of how far back the fact is.
   - Memory horizon: compute each head's effective half-life from its learned decay gates; compare with measured recall.
   - Capacity: how many key-value pairs fit before recall collapses, vs head dim and number of heads.
   - Demo: a CLI that streams R7 over 1M tokens without truncation.
2. **Improving pure-GDN recall**, two debug-scale runs only: larger state (more heads / bigger head dim) and decay-gate bias init toward long memory. A negative result still says something.
3. **Attention sinks and massive activations:** a paragraph if it shows up for free in R2's checkpoints.

Dropped: context-extension method comparisons (its own paper) and circuit-formation interp across checkpoints (checkpoints are saved; analysis later).

## Post-training

One short SFT pass on public chat data for R7 only, so the long-context demo is a chat model. Loss only on assistant turns; hold out ~5%; stop when validation loss turns up. Distillation from a larger open model is a separate experiment, if ever.

## Evaluation

Every eval also runs on the original GPT-2 small weights, so the baseline is measured, not remembered. lm-eval-harness for standard benchmarks (wiring automated, scoring code reviewed); own code for recall tasks.

**Language modeling:** bits per byte on each held-out source. Bits per byte, not loss, because different tokenizers make losses incomparable.

**Benchmarks meaningful at 350M:** HellaSwag, ARC-Easy, PIQA, LAMBADA, WinoGrande, plus ARC-Challenge and a small MMLU slice to see if anything moves above chance. HumanEval still sits near chance; report but don't read into it.

**Long context (question 1).** Needle-in-a-haystack tests verbatim copying only; the point is whether the model integrates, tracks and chains things across the input. Everything below is plotted against distance and, for R2/R6/R7, against model size.

- **Loss vs position:** per-token loss as context grows; a plateau shows where context stops being used, before any benchmark does.
- **Distance-conditioned loss on real data (headline metric):** in held-out code, bucket tokens by distance to the definition they depend on (function defined early, called late; repeated identifiers) and plot loss per bucket; same for repeated entities in long FinePDFs documents. A real-data version of recall vs distance, and what the long-document sources are in the mix for.
- **Context ablation:** loss with full context vs the last k tokens only; the gap is what the far past contributes.
- **State tracking:** RULER variable tracking (chains of reassignments, ask the final value). Tests overwriting old facts, not just storing them.
- **Aggregation:** RULER common/frequent-words tasks (count or rank items across the whole input). A running-summary state may do better here than retrieval.
- **Multi-hop chaining:** BABILong (bAbI ordering, counting, deduction and coreference embedded in long distractor text). Learnable at 350M, so it measures the architecture, not the ceiling.
- **Retrieval without lexical overlap:** NoLiMa-style needles whose question shares no words with the fact. Expect partial results.
- **Baselines:** MQAR and plain needle at 4k/16k for continuity with the literature; one real-task suite (LongBench v2 or HELMET) reported, not interpreted, since it sits near chance at this size.

**Expert prefetch (question 3).** Top-k prediction accuracy of layer l+1's experts from layer l's hidden state (a small probe); routing agreement between neighboring tokens and between consecutive layers, reported as a cache hit rate at k = 1, 2, 4; loss when experts are served from the prediction/cache instead of the true router. Load spread and expert specialization over training for the balancing comparison.

**Interp: deferred.** Log-spaced checkpoints from every run are kept so MLA-latent, routing and decay-gate analyses can be done later at matched points.

## Compute budget

Roughly $450–550 on H100 SXM instances, including slack. Multi-GPU instances run about $1.7–2.2 per GPU-hour; the table uses $2.

| Item | GPU-hours | Cost at $2/GPU-h |
|---|---|---|
| Debug runs, LR sweeps, seeds, search loop (~20–50M params) | ~30–50 | ~$60–100 |
| Ladder R2 (2 seeds), R6, R7, R4 | ~140–160 | ~$280–320 |
| Failed-run slack (+30% on the ladder) | ~45 | ~$90 |
| Long-context evals, SFT on R7 | <5 | <$10 |
| A promoted search-loop run, if any | ~25–30 | ~$50–60 |

Hardware: one 4×H100 SXM node per ladder run (NVLink), on spot with checkpoints every ~30 min; several runs in parallel on separate nodes. Multi-GPU efficiency measured before any 20B run. Runs take hours to a day hands-off, so they overlap with other work.

Sanity check: R2 should reach roughly 150–200k tokens/s per GPU at 4k context, and > 90% of that per GPU at 4 GPUs, both predicted by the performance model first. Measure in the first minutes of every run.

Disk: ~150 GB per instance. Tokenize on a CPU box, not the GPU instance.

## Phases

Each phase gates the next; nothing reaches a 20B run until it works at debug scale and scales efficiently. No dates: apply when the work is ready.

- [ ] 1. Tokenizer and data: BPE encoder (trainer time-boxed), download and filter the sources, uint16 shards on CPU, per-source validation sets. Data filters reviewed by hand.
- [ ] 2. R2 end to end on one GPU: Muon + WSD, eval harness (bits per byte, MQAR, lm-eval), resume from checkpoint; hit the single-GPU MFU target; performance model v1 checked against it.
- [ ] 3. Distributed: DDP and ZeRO-1 by hand, scaling-efficiency table, performance model extended to multi-GPU. Blog post 1: the stack, predicted vs measured.
- [ ] 4. GDN-2: naive version tested against NVlabs code, Triton chunked kernel vs FLA, R6 and R7 at debug scale, GDN-2 MQAR result reproduced.
- [ ] 5. Debug-scale ladder: every architecture at ~20M and ~60M; per-architecture LR sweeps; R2 seed noise floor; capped search loop; profile each architecture.
- [ ] 6. Ladder at 20B: R2 (2 seeds), R6, R7 with the 16k phase; evals with CIs; scaling curves. Blog post 2: results.
- [ ] 7. R4: MLA + MoE, CUDA grouped-GEMM extension, expert parallelism vs data parallel, 20B run.
- [ ] 8. Long context: side investigation 1 on R7/R6, SFT on R7, streaming CLI demo. Blog post 3: very long context with pure linear attention.
- [ ] 9. Numerics: R2 on Blackhole vs H100, FP8 / block-float study.

## What is hand-written vs automated

**By hand:** every model and layer, the tokenizer encoder, training loop, Muon, WSD, DDP/ZeRO-1/expert-parallel, the Triton and CUDA kernels, the performance model, eval scoring for recall tasks.

**Automated (reviewed, not written):** cloud launch, monitoring and checkpoint-sync scripts; data download and sharding plumbing; lm-eval wiring; bootstrap-CI and plotting code; results page and demo. Data filters and eval scoring are the two automated pieces that get a close read, since a bug in either silently changes every comparison.

## Outputs

- Open-source repo: every architecture, the single- and multi-GPU training loop, kernels, tokenizer and eval harness.
- Results page: the ladder table with confidence intervals, scaling curves, predicted-vs-measured performance table, scaling-efficiency table.
- Released weights for the four ladder models.
- Three blog posts: the stack and performance model; ladder results; very long context with pure linear attention.

## References

- DeepSeek-V3 technical report: MLA, DeepSeekMoE, aux-loss-free balancing
- DeepSeek-V4 technical report: CSA, HCA (not in the ladder)
- Gated DeltaNet-2 and NVlabs/GatedDeltaNet-2 code
- Kimi K3 architecture notes (Raschka)
- nanochat: minimal end-to-end tokenizer → pretrain → SFT reference
- Nemotron pre-training datasets collection: Nemotron-CC-v2.1, CC-Code, CC-Math
- Nemotron-CC paper: comparison against DCLM and FineWeb-Edu
- open-sci-ref-0.01: matched-recipe dataset ranking across scales
- Olmo 3 report: Dolma 3 mix, Stack-Edu, long-context data
- H100 pricing snapshot (CloudZero)
