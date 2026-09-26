# Radeon crash investigation handoff — 2026-09-16

Project on remote: `/home/boop/ml/self-play-chess`.
Host: `boop@192.168.4.61` (the name `fractal` timed out from the Mac).
Python 3.14.7, tinygrad 0.14.0, chess 1.11.2, numpy 2.5.3 are installed in
the project's remote `.venv`. A separate local tinygrad checkout is available
at `/home/boop/ml/tinygrad`; it was not modified or installed into this venv.

## User intent

Train two chess transformers from scratch using legal moves and terminal
win/draw/loss outcomes only. No human games, engine labels, material rewards,
or pretrained weights. Stockfish is evaluation-only. Compare end-to-end legal
moves/sec, inference positions/sec, and optimizer throughput across an Apple
M4 Pro (48 GB) and Radeon RX 9070 XT (16 GB). Diagnose the Radeon tuning crash
before trying more aggressive tuning on the user's desktop GPU.

## Current implementation

- Board/state transformer; four independent castling inputs.
- Two learners, alternating colors, batched waves of parallel games.
- TinyJit inference, recreated after each learner update.
- `BEAM=2` affects inference. `TRAIN_BEAM=0` is intentionally the default:
  searching backward kernels had a large startup cost. `TRAIN_BEAM=2` opts in.
- Adam policy-gradient + value cross-entropy + entropy; gradient norm clipping.
- Frozen policies within a wave; one sampled batch/optimizer step per learner
  after completed games. Cutoffs are not draws and are not learning targets.
- Local dashboard, checkpoints including Adam/RNG/config, PGNs, evaluation.
- Tests cover legal moves, gradients, checkpoint restore, changing-input JIT
  equivalence, and parallel move advancement.

## What succeeded on the Radeon

All 11 tests passed:

```sh
cd ~/ml/self-play-chess
DEV=AMD .venv/bin/python -m unittest discover -s tests -v
```

Batch-4 inference measurements (median of 15 warm iterations, transfers included,
encoding excluded), same tinygrad version and model on both machines:

| Mode | M4 Pro positions/sec | Radeon positions/sec |
| --- | ---: | ---: |
| Eager, BEAM=0 | 71.1 | 57.4 |
| TinyJit, BEAM=0 | 782.9 | 505.2 |
| TinyJit, BEAM=2 | 850.8 | **Failed; no timing** |

These tiny models have **18,772 parameters each** (width 32, one layer, four heads,
policy width 8). This workload is not proof of GPU saturation, and these results
do not predict large-model performance. Mac inference tuning's initial warm-up
took about 80 seconds; cached runs were much faster.

Mac warm aggregate rollouts: 8 parallel games ~301 legal moves/sec; 32 ~806.
The verified 32-game run disables learning during rollout timing to avoid
changing weights/invalidating captures. Optimizer batch 8 took ~1.19 seconds,
about 6.74 samples/sec, measured separately with disposable synthetic targets.
Do not confuse rollout throughput with sustained learning throughput.

Raw results are in `runs/performance/`. `mac-rollout-verified.jsonl` supersedes
the earlier 32-game row in `mac-rollout.jsonl`, which mixed warm-up with a learning
update. `profile_run.py` and `throughput.py` reproduce the measurements.

## Failing command and evidence

The remote failure occurred during:

```sh
DEV=AMD BEAM=2 PARALLEL=4 .venv/bin/python -m self_play_chess.profile_run --batch 4 --output runs/performance/amd.jsonl
```

It was warming inference, not updating weights. The traceback passed through
`tinygrad/codegen/opt/search.py:beam_search` → `_time_program` →
`tinygrad/engine/realize.py:time_call` → AMD `invalidate_caches`/`synchronize`.
The final error was:

```
RuntimeError: Wait timeout: 30000 ms! (the signal is not set to 30, but 29)
```

The repeated traceback was extremely large; avoid dumping it unbounded. A full
standalone stderr file was not saved during that attempt. Read driver logs or
redirect any future reproduction to a file after considering desktop disruption.

Driver log around 06:40 on Sep 16:

```
06:40:48 amdgpu 0000:03:00.0: MES might be in unrecoverable state, issue a GPU reset
06:40:48 amdgpu 0000:03:00.0: GPU reset begin!. Source: 3
06:40:52 amdgpu 0000:03:00.0: MODE1 reset
06:40:53 amdgpu 0000:03:00.0: GPU reset succeeded, trying to resume
06:40:53 amdgpu 0000:03:00.0: VRAM is lost due to GPU reset!
06:40:53 amdgpu 0000:03:00.0: GPU reset(1) succeeded!
```

PCI `03:00.0` is the discrete Navi 48 Radeon, while `12:00.0` is the CPU's
Granite Ridge integrated graphics. The reset therefore affected the discrete
card. The user observed a desktop/application crash. No more Radeon workloads
were run after this failure. A read-only process check found no leftover chess
benchmark process. No manual GPU reset or reboot was performed.

This establishes a BEAM candidate/runtime timeout, **not its root cause**.
Potential investigation: compare the pinned pip runtime with the local checkout,
inspect the selected kernel and AMD driver/runtime path, and isolate candidate
search versus normal replay. Do not assume every BEAM=2 kernel or the model
itself is broken. Do not automatically rerun the disruptive command.

## Mac run

The Mac resumed its saved weights with 32 parallel games, BEAM=2 inference,
TinyJit, and zero artificial move delay. A fresh 30-minute monitor was started
after optimization. Its absolute deadline is in the Mac run's `monitor.json`
(scheduled stop 07:13:17 America/New_York on Sep 16). It is evaluating frozen
checkpoints on CPU against official Stockfish and will stop/save at the deadline.
Local dashboard: `http://127.0.0.1:8765` **on the Mac**, not the remote host.

The remote copied run is a snapshot, not a live mirror. Its `monitor.json` is
historical Mac metadata; do not use it to start a remote stop timer. No remote
trainer or monitor was started. Resume remote training only when ready:

```sh
DEV=AMD BEAM=0 .venv/bin/python -m self_play_chess.train --run-dir runs/mac-metal-small --resume --parallel-games 8 --move-delay 0
```

The remote directory excludes the Mac virtualenv and Stockfish binary/source.
The remote virtualenv was created separately; Stockfish would need a Linux build.
