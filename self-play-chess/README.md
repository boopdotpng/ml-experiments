# self-play-chess

A randomly initialized **tinygrad** chess transformer, with **python-chess**
(`import chess`) owning legal moves and game state. Includes two-model self-play
training and a localhost live board. No pretrained model, human game dataset,
engine teacher, or search. Stockfish is used only for evaluation.

## Run

From this directory:

```sh
uv venv
uv pip install -e .
DEV=METAL .venv/bin/python -m unittest discover -s tests -v
DEV=METAL .venv/bin/python -m self_play_chess.train
```

Open **http://127.0.0.1:8765**. The viewer shows live moves, player identities,
outcome predictions, results, timing, and losses. Pause/resume and stop/save are
available in the browser. Stopping during a game discards that unfinished
trajectory and saves weights; stopping during an update waits for it to finish.
The web server stays available after training stops; Ctrl-C shuts it down.

tinygrad 0.14 uses **`DEV=METAL`**, not `DEVICE=METAL` or `METAL=1`.
Use `DEV=CPU` for the CPU fallback. The web server binds only to 127.0.0.1.
The first forward/backward passes compile kernels and can take longer.

For batched games and captured Metal inference:

```sh
DEV=METAL BEAM=2 .venv/bin/python -m self_play_chess.train --parallel-games 8 --move-delay 0 --updates-per-round 8
```

`--parallel-games`, `--move-delay`, `--batch-size`, `--outcome-stratified`, and
`--updates-per-round` can also override
saved settings with `--resume`. Each wave freezes both learners until its games
finish, then samples decisions for the requested number of optimizer steps.
TinyJit captures fixed-shape inference per model; finished slots are padded and
ignored. Captures are recreated after learning updates so weights cannot go stale.
The viewer follows an unfinished board and reports aggregate rollout throughput.
Metal runs on the main thread; HTTP requests use a separate server thread.

`BEAM=2` tunes inference kernels. Training uses `TRAIN_BEAM=0` by default because
tuning all backward kernels has a large startup cost; `TRAIN_BEAM=2` opts in.
Beam-search compilation time is excluded from steady-state benchmark results.

```sh
# Fixed number of games with the small Mac defaults:
DEV=METAL .venv/bin/python -m self_play_chess.train --games 10
# Larger model: 846,532 parameters per learner (verified forward/backward on Metal).
DEV=METAL BEAM=2 .venv/bin/python -m self_play_chess.train --width 128 --layers 4 --heads 4 --policy-width 32 --batch-size 8 --parallel-games 16 --updates-per-round 32 --outcome-stratified --move-delay 0 --run-dir runs/mac-metal-846k
# Resume a previous run (saved architecture and learning settings take precedence):
DEV=METAL .venv/bin/python -m self_play_chess.train --run-dir runs/YOUR-RUN --resume
# Original short untrained demo:
DEV=METAL .venv/bin/python -m self_play_chess --plies 8 --seed 0
```

## Learning loop

Two independently initialized networks, A and B, alternate colors each game.
Weights remain fixed throughout the game. Each selects a move by sampling its
masked policy at temperature 1. There is no teacher and no material reward.
At a genuine terminal result each player gets +1 for winning, 0 for drawing,
or -1 for losing. The value target is loss/draw/win from that player's perspective.

Each step processes 8 decisions by default (`--batch-size` changes this). `--updates-per-round 8` takes eight Adam steps
per learner, using 64 sampled decisions per learner per wave. Sampling is without
replacement when enough decisions exist, otherwise with replacement for fixed
GPU batch shapes. The original default of one step remains for compatibility.

For better experience usage on the small model, the current experiment uses
`--parallel-games 16 --batch-size 8 --updates-per-round 32 --outcome-stratified`:
512 sampled positions per wave across both learners, versus 128 previously.
Stratification includes every available loss/draw/win class in each minibatch.
Each loss term (policy, value, entropy) is weighted by the class's fraction of
rollout positions divided by its fraction of sampled positions. This preserves
the uniform-position objective in expectation rather than changing draw rewards.
Within a class, sampling is without replacement inside each minibatch when
possible; minibatches can reuse positions. All-draw waves still provide no
winning examples. Logs record available positions and decisive samples.

Before any step, behavior-policy log probabilities and value advantages are
frozen for **all** selected minibatches. Later steps use a PPO clipped ratio
against that frozen policy, with clipping range 0.2:

```
advantage = stop_gradient(result - old_expected_value)
ratio = exp(current_log_probability - old_log_probability)
policy_loss = -mean(min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage))
loss = policy_loss + 0.5 * outcome_cross_entropy - 0.01 * policy_entropy
```

Gradients are clipped to global norm 1. The default training model uses width
32, one layer, four heads, and policy width 8; the original model defaults
remain unchanged. This uses Monte Carlo terminal returns and PPO-style policy
clipping, without MCTS, GAE, or repeated epochs over the same minibatch. Early
random play is usually poor, long, and draw-heavy;
the UI's predictions are uncalibrated model outputs, not engine evaluations.
Progress is not guaranteed and self-play win rates are not an Elo measurement.
The small defaults were verified on Metal. A width-64/two-layer configuration
with batch size 32 hit a Metal GPU hang during backward on this Mac; increase
sizes cautiously and verify actual training, not just inference. Batch 32 also
hit a Metal GPU hang with the width-32 model in a disposable training test;
the active experiment keeps the verified batch size of 8.

Games normally run to completion with no arbitrary ply cutoff. If you supply
`--max-plies`, unfinished games are recorded as `*` and **not used for updates**.
`--games 0` (default) runs until stopped. `--move-delay` sets a minimum interval
between moves, not additional delay on top of slow inference.

Each run saves `games.pgn`, `metrics.jsonl`, and an atomically replaced
`latest.safetensors`. The checkpoint contains both models, Adam buffers,
configs, counters, and NumPy RNG state. Resume restarts at a game boundary;
it does not recover an interrupted game's board. Existing checkpoints are
never silently overwritten by a fresh run. Run output is gitignored.

## Input

Fixed white-oriented board: raw `[B, 8, 8]`, flattened to `[B, 64]` in
`a1, b1, ..., h8` order. Piece IDs select embeddings, not numeric piece values:
empty = 0; white pawn/knight/bishop/rook/queen/king = 1–6; black = 7–12.
Learned rank and file embeddings provide coordinates.

One state token combines turn and en-passant square embeddings with a projection
of eight explicit float features:

| Indices | Features |
| --- | --- |
| 0–3 | White kingside, white queenside, black kingside, black queenside rights |
| 4 | Halfmove clock / 150, capped at 1 for terminal inputs |
| 5 | Current position occurrence count / 5, capped at 1 |
| 6 | Threefold draw claim available, including by an intended move |
| 7 | Fifty-move draw claim available, including by an intended move |

Castling is `[1, 1, 1, 1]` initially: **four independent inputs, not a packed
integer**. En passant uses square IDs 0–63 or 64 for none. The resulting tensor
has shape `[B, 65, width]`.

**History limitation:** the environment retains full supplied move history, but
the network sees only the current repetition count and claim availability.
It cannot distinguish every history with different future repetition options.
This is a partially observed baseline, not a complete Markov-state encoding.
A board loaded from FEN cannot recover repetition history that was not supplied.
Future work can expose previous position records to the model.

## Network and actions

Defaults: width 128, 4 layers, 4 attention heads, feed-forward expansion 4.
Pre-normalized residual blocks use full bidirectional attention and GELU MLPs.
All weights start randomly; no language-model checkpoint is involved.

The policy projects each contextual square to source/destination vectors for
five promotion categories: none, knight, bishop, rook, queen. Their dot products
score `[B, 5, 64, 64]` actions. This includes unused slots, masked out by the rules.

```
move ID = promotion_index * 4096 + from_square * 64 + to_square
draw-claim ID = 20480
```

Castling uses standard king source/destination squares. En passant is an ordinary
source/destination action. A separate state-token head scores a draw claim.
Illegal logits become negative infinity before softmax; `ChessEnv.step` also
checks legality before mutation. Terminal positions have no policy and are
rejected by `legal_policy`.

The value head emits three logits: **loss, draw, win for the side to move**.
Expected outcome is `P(win) - P(loss)`. Self-play targets come only
from terminal outcomes, with signs aligned to each saved position's player.

## Rules and scope

Only standard chess is supported. python-chess handles castling, en passant,
promotions, checkmate, stalemate, insufficient material, automatic fivefold
repetition and 75-move draws. Threefold and 50-move draws are optional claim
actions; when a claim relies on an intended move the library validates that
such a move exists. No clocks, resignation, or negotiated draw offers yet.
python-chess's insufficient-material detector does not prove every possible
FIDE dead position; termination follows the library's supported adjudication.

Tests cover encoding, both colors' castling, en passant, promotions, illegal
action rejection, repetition history, draw claims, terminal rewards, batch
forward passes, legal masking, and a finite-gradient optimizer update.
The arbitrary labels in that optimizer test are solely a plumbing check.

Files: `environment.py` (rules/actions), `encoding.py` (observations), `model.py`
(network), `training.py` (learning/checkpoints), `train.py` (local server/CLI),
`static/index.html` (live viewer), and `__main__.py` (untrained rollout).

API references: [tinygrad](https://docs.tinygrad.org/nn/) and
[python-chess](https://python-chess.readthedocs.io/en/stable/core.html).

## Performance and strength benchmarks

```sh
# Same commands on the Radeon machine: replace DEV=METAL with DEV=AMD.
DEV=METAL BEAM=2 .venv/bin/python -m self_play_chess.profile_run --batch 4 --output runs/performance/inference.jsonl
DEV=METAL BEAM=2 .venv/bin/python -m self_play_chess.throughput --parallel-games 8 --output runs/performance/rollout.jsonl
# Match architecture and batches when comparing scaled models:
DEV=METAL BEAM=2 .venv/bin/python -m self_play_chess.throughput --width 128 --layers 4 --heads 4 --policy-width 32 --parallel-games 16 --batch-size 8 --output runs/performance/large-rollout.jsonl
# Kernel timeline/top kernels (PROFILE writes tinygrad's default profile file):
DEV=METAL BEAM=2 PROFILE=1 .venv/bin/python -m self_play_chess.profile_run --batch 4
.venv/bin/python -m tinygrad.viz.cli --interval measure-start measure-end -t 12
```

`profile_run` measures inference including output transfer, excluding board
encoding. `throughput` measures warm aggregate legal moves/sec including rule
handling, encoding, sampling, rendering, and checkpoint work, followed by
optimizer samples/sec on **disposable synthetic targets**. It never modifies real
run weights. Neither metric proves full GPU utilization. Compare batch sizes
and report warm-up costs separately. Full training also includes game completion
and learning updates; short capped rollouts alone are not a learning benchmark.

Stockfish is evaluation-only. To build its official source on Apple Silicon:

```sh
git clone --depth 1 https://github.com/official-stockfish/Stockfish.git vendor/Stockfish
make -C vendor/Stockfish/src -j4 build ARCH=apple-silicon COMP=clang
DEV=CPU .venv/bin/python -m self_play_chess.benchmark --run-dir runs/YOUR-RUN --stockfish vendor/Stockfish/src/stockfish --minutes 30
```

The monitor evaluates frozen copies on CPU against random legal play, depth-1
Stockfish, a 100-node budget, and the engine's minimum requested Elo at 10ms/move.
Every model plays both colors. Seeds and engine identity/settings are recorded.
Two games per condition are only a smoke benchmark, not a reliable rating.
Cutoffs are reported separately and excluded from score. Engine-requested Elo
at this budget is not a calibrated estimate of the learner's Elo.
Results go to `benchmarks.jsonl` and the viewer's `evaluation.json`. A separate
timer stops the local trainer after 30 minutes, then evaluates its final checkpoint.
