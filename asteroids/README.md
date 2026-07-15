# Asteroids PPO

A compact PyTorch agent that learns `ALE/Asteroids-v5` directly from pixels. Gymnasium/ALE is only the simulator; the CNN, rollout buffer, GAE, PPO update, reward shaping, logging, evaluation, and checkpoints are implemented in [`asteroids.py`](asteroids.py).

This directory now contains two agents:

- [`asteroids.py`](asteroids.py): the small 1.7M-parameter PPO baseline.
- [`scaled_q.py`](scaled_q.py): the serious 29.9M-parameter recurrent distributional Q-agent described below.

## Scaled recurrent agent

The scaled agent uses an IMPALA-style residual CNN, 1024-unit LSTM, dueling implicit-quantile heads, NoisyNet exploration, n-step Double-Q targets, prioritized recurrent replay, and 24 parallel ALE actors. Its two value heads independently predict transformed game score and future life losses. Forty-eight replay sequences are learned together in one large GPU update. This measured about 9,500 emulator frames/s while using 26.6 GB of the 5090's VRAM. Actions maximize:

```text
expected score + 0.01 per surviving step - 20 × expected future life losses
```

The risk weight ramps from zero to 20 over the first five million frames so the score policy has time to form before survival pressure reaches full strength. It can also be changed during evaluation without retraining. Compact replay stores one new grayscale frame per step rather than four overlapping frame stacks, so the default 4,096-sequence buffer occupies about 2.4 GB rather than roughly 9.5 GB. Greedy evaluation runs automatically every million frames; the best score and survival policies are retained separately, with archival snapshots every five million frames.

Inspect the two provided model scales:

```bash
.venv/bin/python asteroids/scaled_q.py model-info
.venv/bin/python asteroids/scaled_q.py model-info --width 4 --hidden 2048
```

They contain 29.9M and 119.2M parameters respectively. Start with 29.9M: scaling to 119M multiplies inference and replay-learning cost, and should be justified by an ablation rather than faith in parameter count.

Train and evaluate the default model:

```bash
.venv/bin/python asteroids/scaled_q.py train --total-frames 200000000
.venv/bin/python asteroids/scaled_q.py eval \
  asteroids/runs/scaled/scaled_latest.pt --episodes 20 --video asteroids/runs/scaled/play.mp4
```

Test different survival tradeoffs on the same checkpoint:

```bash
.venv/bin/python asteroids/scaled_q.py eval asteroids/runs/scaled/scaled_latest.pt --risk-weight 10
.venv/bin/python asteroids/scaled_q.py eval asteroids/runs/scaled/scaled_latest.pt --risk-weight 40
```

For the intentionally oversized version, use a separate run:

```bash
.venv/bin/python asteroids/scaled_q.py train --width 4 --hidden 2048 \
  --batch-sequences 4 --output asteroids/runs/scaled-119m
```

The environment runs fully headless. Observations use the standard Atari recipe: random no-op reset, pixel max across a 4-frame action repeat, 84x84 grayscale, and a history of four frames. The v5 environment retains its 25% sticky-action probability.

## Train

From the repository root:

```bash
uv pip install --python .venv/bin/python -r asteroids/requirements.txt
.venv/bin/python asteroids/asteroids.py train
```

The defaults are intended for this machine: 8 ALE actors, CUDA, 50 million emulator frames, signed-square-root score rewards, and a `-5` penalty for losing a life. Progress goes to `asteroids/runs/default/train.csv`; interrupting with Ctrl-C safely writes `latest.pt`.

A useful first run is 10 million frames. For a serious result, let the default 50 million finish, inspect the learning curve, then continue toward 100-200 million:

```bash
.venv/bin/python asteroids/asteroids.py train --total-frames 10000000 --output asteroids/runs/10m
.venv/bin/python asteroids/asteroids.py train --total-frames 100000000 \
  --output asteroids/runs/10m --resume asteroids/runs/10m/latest.pt
```

`--total-frames` is the final total, not the number of extra frames after resuming. Run `train --help` to see the small set of tuning knobs.

## Evaluate and record

Evaluation chooses the highest-probability action and leaves sticky controls enabled:

```bash
.venv/bin/python asteroids/asteroids.py eval asteroids/runs/10m/latest.pt --episodes 20
.venv/bin/python asteroids/asteroids.py eval asteroids/runs/10m/latest.pt \
  --episodes 5 --video asteroids/runs/10m/play.mp4
```

MP4 capture also works headlessly. While the policy is still highly exploratory early in training, `--sample-actions` measures the stochastic policy PPO actually optimized. Scores are always native game scores; only the reward seen during training is transformed.

## Why PPO?

Rainbow/R2D2 can be more sample-efficient, but recurrent prioritized sequence replay and categorical Q projections add a great deal of code and several subtle failure modes. Parallel PPO is robust, fits in one readable file, and uses the 5090 for large batched updates. Four stacked frames expose velocity, while the explicit life penalty biases the policy toward the requested long-survival behavior without removing the incentive to score.
