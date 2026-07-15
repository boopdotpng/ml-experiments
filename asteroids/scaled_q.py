"""A scalable recurrent distributional Q-agent for Asteroids.

This is deliberately a single-file implementation of the agent side of RL:
an IMPALA-style visual encoder, LSTM memory, dual IQN score/risk heads,
prioritized sequence replay, n-step Double-Q targets, parallel actors,
checkpointing, and evaluation. Gymnasium/ALE is only the simulator.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
import random
import time
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from asteroids import FRAME_SKIP, STACK_SIZE, episode_stats, final_lives, make_env, signed_sqrt


def make_vector_env(args: argparse.Namespace, generation: int = 0) -> gym.vector.AsyncVectorEnv:
    seed_base = args.seed + generation * 100_000
    return gym.vector.AsyncVectorEnv(
        [partial(make_env, seed_base + i) for i in range(args.num_envs)],
        shared_memory=True,
        # CPython 3.14 defaults to forkserver on Linux. Explicit spawn keeps
        # ALE/OpenCV native state out of a forking server process.
        context="spawn",
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def force_close_vector(envs: gym.vector.AsyncVectorEnv) -> None:
    """Close an actor pool even when a native ALE worker has crashed mid-step."""
    try:
        envs.close(terminate=True)
        return
    except Exception:
        pass
    for process in envs.processes:
        if process.is_alive():
            process.terminate()
    for process in envs.processes:
        process.join(timeout=2)
    for pipe in envs.parent_pipes:
        try:
            pipe.close()
        except Exception:
            pass


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer; deterministic when the model is in eval mode."""

    def __init__(self, inputs: int, outputs: int, sigma: float = 0.5) -> None:
        super().__init__()
        self.inputs, self.outputs = inputs, outputs
        bound = 1 / math.sqrt(inputs)
        self.weight_mu = nn.Parameter(torch.empty(outputs, inputs).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(torch.full((outputs, inputs), sigma / math.sqrt(inputs)))
        self.bias_mu = nn.Parameter(torch.empty(outputs).uniform_(-bound, bound))
        self.bias_sigma = nn.Parameter(torch.full((outputs,), sigma / math.sqrt(outputs)))
        self.register_buffer("weight_epsilon", torch.zeros(outputs, inputs))
        self.register_buffer("bias_epsilon", torch.zeros(outputs))
        self.reset_noise()

    @staticmethod
    def scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self) -> None:
        epsilon_in = self.scaled_noise(self.inputs, self.weight_mu.device)
        epsilon_out = self.scaled_noise(self.outputs, self.weight_mu.device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual


class ImpalaStage(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.blocks = nn.Sequential(ResidualBlock(outputs), ResidualBlock(outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        return self.blocks(x)


class IQNHead(nn.Module):
    """Dueling implicit quantile head for one return signal."""

    def __init__(self, hidden: int, actions: int, num_cosines: int = 64) -> None:
        super().__init__()
        self.hidden, self.actions, self.num_cosines = hidden, actions, num_cosines
        self.cosine = nn.Linear(num_cosines, hidden)
        self.value1, self.value2 = NoisyLinear(hidden, hidden), NoisyLinear(hidden, 1)
        self.advantage1, self.advantage2 = NoisyLinear(hidden, hidden), NoisyLinear(hidden, actions)
        self.register_buffer("frequencies", torch.arange(1, num_cosines + 1).float() * math.pi)

    def reset_noise(self) -> None:
        for module in (self.value1, self.value2, self.advantage1, self.advantage2):
            module.reset_noise()

    def forward(
        self, features: torch.Tensor, num_quantiles: int, taus: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leading = features.shape[:-1]
        flat = features.reshape(-1, self.hidden)
        if taus is None:
            taus = torch.rand(flat.shape[0], num_quantiles, device=features.device)
        cosine = torch.cos(taus.unsqueeze(-1) * self.frequencies)
        embedding = F.relu(self.cosine(cosine))
        x = flat.unsqueeze(1) * embedding
        value = self.value2(F.relu(self.value1(x)))
        advantage = self.advantage2(F.relu(self.advantage1(x)))
        quantiles = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return quantiles.reshape(*leading, num_quantiles, self.actions), taus.reshape(*leading, num_quantiles)


class ScaledAgent(nn.Module):
    """Large visual recurrent model with separate score and life-loss distributions."""

    def __init__(self, actions: int = 14, width: int = 2, hidden: int = 1024, cosines: int = 64) -> None:
        super().__init__()
        channels = [32 * width, 64 * width, 128 * width]
        self.hidden_size = hidden
        self.encoder = nn.Sequential(
            ImpalaStage(STACK_SIZE, channels[0]),
            ImpalaStage(channels[0], channels[1]),
            ImpalaStage(channels[1], channels[2]),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(channels[2] * 6 * 6, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.memory = nn.LSTM(hidden, hidden, batch_first=True)
        self.score_head = IQNHead(hidden, actions, cosines)
        self.risk_head = IQNHead(hidden, actions, cosines)

    def initial_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(1, batch, self.hidden_size, device=device)
        return zeros, zeros.clone()

    def reset_noise(self) -> None:
        self.score_head.reset_noise()
        self.risk_head.reset_noise()

    def visual(self, obs: torch.Tensor) -> torch.Tensor:
        shape = obs.shape[:-3]
        x = self.encoder(obs.reshape(-1, STACK_SIZE, 84, 84).float().div_(255.0))
        return x.reshape(*shape, self.hidden_size)

    def sequence_features(
        self, obs: torch.Tensor, burn_in: int = 0
    ) -> torch.Tensor:
        """Encode [batch,time,4,84,84], stopping gradients through burn-in."""
        if burn_in == 0:
            features, _ = self.memory(self.visual(obs))
            return features
        with torch.no_grad():
            _, state = self.memory(self.visual(obs[:, :burn_in]))
        features, _ = self.memory(self.visual(obs[:, burn_in:]), (state[0].detach(), state[1].detach()))
        return features

    def step_features(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features, state = self.memory(self.visual(obs).unsqueeze(1), state)
        return features[:, 0], state

    def q_values(
        self, features: torch.Tensor, quantiles: int, risk_weight: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score, _ = self.score_head(features, quantiles)
        risk, _ = self.risk_head(features, quantiles)
        score_q, risk_q = score.mean(dim=-2), risk.mean(dim=-2)
        return score_q - risk_weight * risk_q, score_q, risk_q


class SequenceReplay:
    """Prioritized replay storing overlapping frame stacks only once.

    One sequence stores its initial four frames plus one new frame per step.
    This is about four times smaller than storing every stacked observation.
    """

    def __init__(self, capacity: int, sequence_length: int, alpha: float = 0.6) -> None:
        self.capacity, self.sequence_length, self.alpha = capacity, sequence_length, alpha
        self.frames = np.empty((capacity, sequence_length + STACK_SIZE, 84, 84), dtype=np.uint8)
        self.actions = np.empty((capacity, sequence_length), dtype=np.uint8)
        self.rewards = np.empty((capacity, sequence_length), dtype=np.float32)
        self.costs = np.empty((capacity, sequence_length), dtype=np.float32)
        self.dones = np.empty((capacity, sequence_length), dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = self.size = 0

    @property
    def memory_gb(self) -> float:
        return self.frames.nbytes / 2**30

    def add_batch(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        # Inputs are [time(+stack), environment, ...].
        maximum = max(1.0, float(self.priorities[: self.size].max(initial=1.0)))
        for env in range(actions.shape[1]):
            index = self.position
            self.frames[index] = frames[:, env]
            self.actions[index] = actions[:, env]
            self.rewards[index] = rewards[:, env]
            self.costs[index] = costs[:, env]
            self.dones[index] = dones[:, env]
            self.priorities[index] = maximum
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch: int, beta: float, device: torch.device
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        probabilities = self.priorities[: self.size].astype(np.float64) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(self.size, batch, replace=self.size < batch, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        data = {
            "frames": torch.as_tensor(self.frames[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices].astype(np.int64), device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "costs": torch.as_tensor(self.costs[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
        }
        return data, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = np.maximum(priorities, 1e-4)


def stacked_observations(frames: torch.Tensor) -> torch.Tensor:
    """Convert [B,T+4,H,W] compact frames to [B,T+1,4,H,W]."""
    return frames.unfold(1, STACK_SIZE, 1).permute(0, 1, 4, 2, 3)


def n_step_returns(
    rewards: torch.Tensor, dones: torch.Tensor, starts: torch.Tensor, n: int, gamma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return discounted rewards and bootstrap multipliers for arbitrary start times."""
    result = torch.zeros((rewards.shape[0], starts.numel()), device=rewards.device)
    alive = torch.ones_like(result)
    discount = 1.0
    for offset in range(n):
        step = starts + offset
        result += discount * alive * rewards[:, step]
        alive *= 1.0 - dones[:, step]
        discount *= gamma
    return result, alive * discount


def quantile_huber(
    prediction: torch.Tensor, target: torch.Tensor, taus: torch.Tensor
) -> torch.Tensor:
    """Per-item IQN loss for [items,predicted quantiles] and [items,target quantiles]."""
    delta = target.unsqueeze(1) - prediction.unsqueeze(2)
    absolute = delta.abs()
    huber = torch.where(absolute <= 1.0, 0.5 * delta.square(), absolute - 0.5)
    weight = (taus.unsqueeze(2) - (delta.detach() < 0).float()).abs()
    return (weight * huber).mean(dim=(1, 2))


def reset_finished_state(
    state: tuple[torch.Tensor, torch.Tensor], done: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    alive = torch.as_tensor(~done, device=state[0].device).view(1, -1, 1)
    return state[0] * alive, state[1] * alive


def collect_sequences(
    envs: gym.vector.VectorEnv,
    agent: ScaledAgent,
    obs_np: np.ndarray,
    last_lives: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    frames_seen: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[tuple[float, int]]]:
    length, count = args.sequence_length, args.num_envs
    compact_frames = np.empty((length + STACK_SIZE, count, 84, 84), dtype=np.uint8)
    compact_frames[:STACK_SIZE] = obs_np.transpose(1, 0, 2, 3)
    actions = np.empty((length, count), dtype=np.uint8)
    rewards = np.empty((length, count), dtype=np.float32)
    costs = np.empty((length, count), dtype=np.float32)
    dones = np.empty((length, count), dtype=np.bool_)
    completed: list[tuple[float, int]] = []
    state = agent.initial_state(count, device)
    agent.train()  # Enable NoisyNet exploration.

    for step in range(length):
        progress = min(1.0, frames_seen / max(1, args.exploration_frames))
        epsilon = args.epsilon_start + progress * (args.epsilon_final - args.epsilon_start)
        agent.reset_noise()
        with torch.inference_mode():
            obs = torch.as_tensor(obs_np, dtype=torch.uint8, device=device)
            features, state = agent.step_features(obs, state)
            utility, _, _ = agent.q_values(
                features, args.actor_quantiles, scheduled_risk_weight(frames_seen, args)
            )
            action = utility.argmax(dim=-1).cpu().numpy()
        random_mask = np.random.random(count) < epsilon
        action[random_mask] = np.random.randint(0, utility.shape[-1], random_mask.sum())

        # A dead actor must become a visible, checkpointed failure rather than an
        # infinite wait inside VectorEnv.step().
        envs.step_async(action)
        next_obs, raw_reward, terminated, truncated, infos = envs.step_wait(timeout=args.actor_timeout)
        done = np.logical_or(terminated, truncated)
        lives_before_reset = final_lives(infos, done)
        life_lost = lives_before_reset < last_lives
        actions[step] = action
        rewards[step] = signed_sqrt(raw_reward) + args.alive_bonus
        costs[step] = life_lost
        dones[step] = done
        compact_frames[step + STACK_SIZE] = next_obs[:, -1]
        completed.extend(episode_stats(infos))
        state = reset_finished_state(state, done)
        obs_np = next_obs
        last_lives = np.asarray(infos["lives"])
        frames_seen += count * FRAME_SKIP
    return obs_np, last_lives, {"frames": compact_frames, "actions": actions, "rewards": rewards, "costs": costs, "dones": dones}, completed


def learn(
    agent: ScaledAgent,
    target: ScaledAgent,
    replay: SequenceReplay,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    beta: float,
    risk_weight: float,
) -> dict[str, float]:
    data, indices = replay.sample(args.batch_sequences, beta, device)
    obs = stacked_observations(data["frames"])
    learn_length = args.sequence_length - args.burn_in - args.n_step
    starts = torch.arange(args.burn_in, args.burn_in + learn_length, device=device)

    agent.train()
    agent.reset_noise()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
        features = agent.sequence_features(obs, args.burn_in)
        current = features[:, :learn_length]
        following_online = features[:, args.n_step : args.n_step + learn_length].detach()
        score_prediction, taus = agent.score_head(current, args.quantiles)
        risk_prediction, risk_taus = agent.risk_head(current, args.quantiles)
        action = data["actions"][:, starts]
        gather = action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.quantiles, 1)
        score_prediction = score_prediction.gather(-1, gather).squeeze(-1)
        risk_prediction = risk_prediction.gather(-1, gather).squeeze(-1)

        with torch.no_grad():
            next_utility, _, _ = agent.q_values(following_online, args.actor_quantiles, risk_weight)
            next_action = next_utility.argmax(dim=-1)
            target_features = target.sequence_features(obs, args.burn_in)
            following_target = target_features[:, args.n_step : args.n_step + learn_length]
            target_score, _ = target.score_head(following_target, args.target_quantiles)
            target_risk, _ = target.risk_head(following_target, args.target_quantiles)
            target_gather = next_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, args.target_quantiles, 1)
            target_score = target_score.gather(-1, target_gather).squeeze(-1)
            target_risk = target_risk.gather(-1, target_gather).squeeze(-1)
            score_return, bootstrap = n_step_returns(data["rewards"], data["dones"], starts, args.n_step, args.gamma)
            risk_return, _ = n_step_returns(data["costs"], data["dones"], starts, args.n_step, args.gamma)
            target_score = score_return.unsqueeze(-1) + bootstrap.unsqueeze(-1) * target_score
            target_risk = risk_return.unsqueeze(-1) + bootstrap.unsqueeze(-1) * target_risk

        items = args.batch_sequences * learn_length
        score_losses = quantile_huber(
            score_prediction.reshape(items, args.quantiles),
            target_score.reshape(items, args.target_quantiles),
            taus.reshape(items, args.quantiles),
        ).reshape(args.batch_sequences, learn_length)
        risk_losses = quantile_huber(
            risk_prediction.reshape(items, args.quantiles),
            target_risk.reshape(items, args.target_quantiles),
            risk_taus.reshape(items, args.quantiles),
        ).reshape(args.batch_sequences, learn_length)
        # Actor state is reset at an episode boundary. The fast cuDNN sequence
        # pass cannot reset in its middle, so do not train on samples after the
        # first boundary in a replay sequence. Terminal transitions themselves
        # remain valid and correctly receive no bootstrap value.
        previous_terminals = data["dones"].cumsum(dim=1) - data["dones"]
        valid = (previous_terminals[:, starts] == 0).float()
        valid_count = valid.sum(dim=1).clamp_min(1.0)
        per_sequence = ((score_losses + args.risk_loss_coef * risk_losses) * valid).sum(1) / valid_count
        loss = (per_sequence * data["weights"]).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        score_td = (target_score.mean(-1) - score_prediction.mean(-1)).abs()
        risk_td = (target_risk.mean(-1) - risk_prediction.mean(-1)).abs()
        priorities = ((score_td + risk_td) * valid).amax(dim=1).float().cpu().numpy()
    replay.update_priorities(indices, priorities)
    metric_count = valid.sum().clamp_min(1.0)
    return {
        "loss": float(loss.detach()),
        "score_loss": float((score_losses * valid).sum().detach() / metric_count),
        "risk_loss": float((risk_losses * valid).sum().detach() / metric_count),
        "grad_norm": float(grad_norm),
        "mean_priority": float(priorities.mean()),
    }


def save_checkpoint(
    path: Path,
    agent: ScaledAgent,
    optimizer: torch.optim.Optimizer,
    frames: int,
    updates: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    config = {key: value for key, value in vars(args).items() if key != "func"}
    torch.save(
        {"model": agent.state_dict(), "optimizer": optimizer.state_dict(), "frames": frames, "updates": updates, "args": config},
        temporary,
    )
    os.replace(temporary, path)


def save_policy(path: Path, agent: ScaledAgent, frames: int, updates: int, args: argparse.Namespace) -> None:
    """Save an evaluation-only snapshot without the large Adam state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key != "func"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": agent.state_dict(), "frames": frames, "updates": updates, "args": config}, temporary)
    os.replace(temporary, path)


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters(), strict=True):
            target_parameter.lerp_(online_parameter, tau)


def scheduled_risk_weight(frames: int, args: argparse.Namespace) -> float:
    return args.risk_weight * min(1.0, frames / max(1, args.risk_warmup_frames))


@torch.inference_mode()
def evaluate_current(
    agent: ScaledAgent, device: torch.device, seed: int, episodes: int, quantiles: int, risk_weight: float
) -> tuple[float, float]:
    """Evaluate between actor rollouts, avoiding a competing CUDA process."""
    env = make_env(seed)
    scores: list[float] = []
    lengths: list[int] = []
    agent.eval()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            state = agent.initial_state(1, device)
            score = steps = 0
            done = False
            while not done:
                tensor = torch.as_tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)
                features, state = agent.step_features(tensor, state)
                utility, _, _ = agent.q_values(features, quantiles, risk_weight)
                obs, reward, terminated, truncated, _ = env.step(utility.argmax(-1).item())
                score += reward
                steps += 1
                done = terminated or truncated
            scores.append(score)
            lengths.append(steps * FRAME_SKIP)
    finally:
        env.close()
        agent.train()
    return float(np.mean(scores)), float(np.mean(lengths))


def build_from_checkpoint(path: Path, device: torch.device) -> tuple[ScaledAgent, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["args"]
    agent = ScaledAgent(width=int(config["width"]), hidden=int(config["hidden"]), cosines=int(config["cosines"])).to(device)
    agent.load_state_dict(checkpoint["model"])
    agent.eval()
    return agent, checkpoint


def train(args: argparse.Namespace) -> None:
    if args.burn_in + args.n_step >= args.sequence_length:
        raise SystemExit("burn-in + n-step must be smaller than sequence-length")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved = checkpoint["args"]
        # Architectural values are intrinsic to the checkpoint; training knobs may change.
        args.width, args.hidden, args.cosines = int(saved["width"]), int(saved["hidden"]), int(saved["cosines"])

    envs = make_vector_env(args)
    agent = ScaledAgent(width=args.width, hidden=args.hidden, cosines=args.cosines).to(device)
    target = ScaledAgent(width=args.width, hidden=args.hidden, cosines=args.cosines).to(device)
    target.load_state_dict(agent.state_dict())
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=args.weight_decay)
    frames_seen = updates = 0
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        frames_seen = int(checkpoint.get("frames", 0))
        updates = int(checkpoint.get("updates", 0))

    replay = SequenceReplay(args.replay_sequences, args.sequence_length, args.priority_alpha)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "scaled_train.csv"
    append = log_path.exists() and bool(args.resume)
    log_file = log_path.open("a" if append else "w", newline="")
    fields = ["frames", "updates", "fps", "gpu_gb", "actor_restarts", "score_100", "length_100", "risk_weight", "epsilon", "loss", "score_loss", "risk_loss", "grad_norm", "priority", "replay"]
    logger = csv.DictWriter(log_file, fieldnames=fields)
    if not append:
        logger.writeheader()
    eval_path = output / "scaled_eval.csv"
    eval_append = eval_path.exists() and bool(args.resume)
    eval_file = eval_path.open("a" if eval_append else "w", newline="")
    eval_logger = csv.DictWriter(eval_file, fieldnames=["frames", "score", "length", "risk_weight"])
    if not eval_append:
        eval_logger.writeheader()

    obs_np, infos = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    last_lives = np.asarray(infos["lives"])
    scores: deque[float] = deque(maxlen=100)
    lengths: deque[int] = deque(maxlen=100)
    start_frames, start_time = frames_seen, time.monotonic()
    parameters = sum(parameter.numel() for parameter in agent.parameters())
    print(f"scaled agent: {parameters / 1e6:.1f}M parameters on {device}; replay capacity {replay.memory_gb:.1f} GB")
    print(f"objective: signed_sqrt(score) + {args.alive_bonus:g}/step - {args.risk_weight:g} * predicted life-loss risk")

    metrics: dict[str, float] = {name: 0.0 for name in ("loss", "score_loss", "risk_loss", "grad_norm", "mean_priority")}
    cycle = 0
    next_eval = (frames_seen // args.eval_every + 1) * args.eval_every
    next_archive = (frames_seen // args.archive_every + 1) * args.archive_every
    best_score = best_length = -math.inf
    actor_restarts = 0
    try:
        while frames_seen < args.total_frames:
            try:
                obs_np, last_lives, sequence, completed = collect_sequences(
                    envs, agent, obs_np, last_lives, args, device, frames_seen
                )
            except (EOFError, BrokenPipeError, multiprocessing.TimeoutError) as error:
                actor_restarts += 1
                print(f"actor pool failed ({type(error).__name__}); restarting pool #{actor_restarts}")
                force_close_vector(envs)
                envs = make_vector_env(args, actor_restarts)
                obs_np, infos = envs.reset(
                    seed=[args.seed + actor_restarts * 100_000 + i for i in range(args.num_envs)]
                )
                last_lives = np.asarray(infos["lives"])
                continue
            replay.add_batch(**sequence)
            frames_seen += args.num_envs * args.sequence_length * FRAME_SKIP
            for score, length in completed:
                scores.append(score)
                lengths.append(length * FRAME_SKIP)

            if replay.size >= args.min_replay_sequences:
                progress = min(1.0, frames_seen / max(1, args.total_frames))
                beta = args.priority_beta_start + progress * (1.0 - args.priority_beta_start)
                accumulated = []
                current_risk = scheduled_risk_weight(frames_seen, args)
                for _ in range(args.updates_per_rollout):
                    accumulated.append(learn(agent, target, replay, optimizer, args, device, beta, current_risk))
                    updates += 1
                    soft_update(target, agent, args.target_tau)
                metrics = {key: float(np.mean([item[key] for item in accumulated])) for key in accumulated[0]}

            cycle += 1
            if cycle == 1 or cycle % args.log_every == 0:
                fps = int((frames_seen - start_frames) / max(time.monotonic() - start_time, 1e-6))
                exploration_progress = min(1.0, frames_seen / max(1, args.exploration_frames))
                epsilon = args.epsilon_start + exploration_progress * (args.epsilon_final - args.epsilon_start)
                row = {
                    "frames": frames_seen,
                    "updates": updates,
                    "fps": fps,
                    "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
                    "actor_restarts": actor_restarts,
                    "score_100": round(float(np.mean(scores)), 1) if scores else "",
                    "length_100": round(float(np.mean(lengths)), 1) if lengths else "",
                    "risk_weight": scheduled_risk_weight(frames_seen, args),
                    "epsilon": epsilon,
                    "loss": metrics["loss"],
                    "score_loss": metrics["score_loss"],
                    "risk_loss": metrics["risk_loss"],
                    "grad_norm": metrics["grad_norm"],
                    "priority": metrics["mean_priority"],
                    "replay": replay.size,
                }
                logger.writerow(row)
                log_file.flush()
                print(f"frames={frames_seen:>11,} fps={fps:>6} gpu={row['gpu_gb']:>5}GB replay={replay.size:>4} score100={row['score_100'] or 'waiting':>7} length100={row['length_100'] or 'waiting':>7} loss={metrics['loss']:.3f}")
            if frames_seen >= next_eval:
                eval_risk = scheduled_risk_weight(frames_seen, args)
                eval_score, eval_length = evaluate_current(
                    agent, device, args.seed + 100_000, args.eval_episodes, args.actor_quantiles, eval_risk
                )
                eval_logger.writerow({"frames": frames_seen, "score": eval_score, "length": eval_length, "risk_weight": eval_risk})
                eval_file.flush()
                print(f"EVAL frames={frames_seen:,} score={eval_score:.1f} length={eval_length:.0f}")
                if eval_score > best_score:
                    best_score = eval_score
                    save_policy(output / "scaled_best_score.pt", agent, frames_seen, updates, args)
                if eval_length > best_length:
                    best_length = eval_length
                    save_policy(output / "scaled_best_survival.pt", agent, frames_seen, updates, args)
                next_eval += args.eval_every
            if frames_seen >= next_archive:
                save_policy(output / f"scaled_frames-{frames_seen}.pt", agent, frames_seen, updates, args)
                next_archive += args.archive_every
            if cycle % args.save_every == 0 or frames_seen >= args.total_frames:
                save_checkpoint(output / "scaled_latest.pt", agent, optimizer, frames_seen, updates, args)
    except KeyboardInterrupt:
        print("interrupted; saving current network (replay is intentionally not checkpointed)")
        save_checkpoint(output / "scaled_latest.pt", agent, optimizer, frames_seen, updates, args)
    except Exception as error:
        print(f"training failed at {frames_seen:,} frames: {type(error).__name__}: {error}")
        save_checkpoint(output / "scaled_latest.pt", agent, optimizer, frames_seen, updates, args)
        raise
    finally:
        # terminate=True is intentional: it also makes cleanup safe if an actor
        # died while an asynchronous step was pending.
        force_close_vector(envs)
        log_file.close()
        eval_file.close()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    agent, checkpoint = build_from_checkpoint(args.checkpoint, device)
    config = checkpoint["args"]
    risk_weight = args.risk_weight if args.risk_weight is not None else float(config["risk_weight"])
    env = make_env(args.seed, render_mode="rgb_array" if args.video else None)
    writer = None
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=15, codec="libx264", quality=8, macro_block_size=2)
    scores, lengths = [], []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            state = agent.initial_state(1, device)
            score = steps = 0
            done = False
            while not done:
                tensor = torch.as_tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)
                features, state = agent.step_features(tensor, state)
                utility, score_q, risk_q = agent.q_values(features, args.quantiles, risk_weight)
                action = utility.argmax(-1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                score += reward
                steps += 1
                done = terminated or truncated
                if writer is not None and episode == 0:
                    writer.append_data(env.render())
            scores.append(score)
            lengths.append(steps * FRAME_SKIP)
            print(f"episode {episode + 1:>3}: score={score:>8.0f}, frames={steps * FRAME_SKIP:>8,}")
    finally:
        env.close()
        if writer is not None:
            writer.close()
    print(f"risk weight={risk_weight:g}; checkpoint frames={int(checkpoint.get('frames', 0)):,}")
    print(f"score mean={np.mean(scores):.1f} median={np.median(scores):.1f} min={np.min(scores):.1f} max={np.max(scores):.1f}")
    print(f"length mean={np.mean(lengths):,.0f} emulator frames")


def model_info(args: argparse.Namespace) -> None:
    model = ScaledAgent(width=args.width, hidden=args.hidden, cosines=args.cosines)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters: {parameters:,} ({parameters / 1e6:.1f}M)")
    print(f"approx fp32 weights: {parameters * 4 / 2**20:.1f} MiB")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    common_model = argparse.ArgumentParser(add_help=False)
    common_model.add_argument("--width", type=int, default=2, help="CNN width multiplier; 4 is intentionally huge")
    common_model.add_argument("--hidden", type=int, default=1024)
    common_model.add_argument("--cosines", type=int, default=64)

    training = commands.add_parser("train", parents=[common_model])
    training.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--total-frames", type=int, default=200_000_000)
    training.add_argument("--num-envs", type=int, default=24)
    training.add_argument("--actor-timeout", type=float, default=30.0)
    training.add_argument("--sequence-length", type=int, default=80)
    training.add_argument("--burn-in", type=int, default=20)
    training.add_argument("--n-step", type=int, default=5)
    training.add_argument("--batch-sequences", type=int, default=48)
    training.add_argument("--replay-sequences", type=int, default=4096)
    training.add_argument("--min-replay-sequences", type=int, default=128)
    training.add_argument("--updates-per-rollout", type=int, default=1)
    training.add_argument("--quantiles", type=int, default=32)
    training.add_argument("--target-quantiles", type=int, default=32)
    training.add_argument("--actor-quantiles", type=int, default=16)
    training.add_argument("--gamma", type=float, default=0.997)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--weight-decay", type=float, default=1e-5)
    training.add_argument("--max-grad-norm", type=float, default=10.0)
    training.add_argument("--target-tau", type=float, default=0.005)
    training.add_argument("--priority-alpha", type=float, default=0.6)
    training.add_argument("--priority-beta-start", type=float, default=0.4)
    training.add_argument("--risk-weight", type=float, default=20.0)
    training.add_argument("--risk-warmup-frames", type=int, default=5_000_000)
    training.add_argument("--risk-loss-coef", type=float, default=1.0)
    training.add_argument("--alive-bonus", type=float, default=0.01)
    training.add_argument("--epsilon-start", type=float, default=0.4)
    training.add_argument("--epsilon-final", type=float, default=0.01)
    training.add_argument("--exploration-frames", type=int, default=10_000_000)
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--output", default="asteroids/runs/scaled")
    training.add_argument("--resume", type=Path)
    training.add_argument("--log-every", type=int, default=10)
    training.add_argument("--save-every", type=int, default=100)
    training.add_argument("--eval-every", type=int, default=1_000_000)
    training.add_argument("--eval-episodes", type=int, default=5)
    training.add_argument("--archive-every", type=int, default=5_000_000)
    training.set_defaults(func=train)

    evaluating = commands.add_parser("eval")
    evaluating.add_argument("checkpoint", type=Path)
    evaluating.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    evaluating.add_argument("--seed", type=int, default=1)
    evaluating.add_argument("--episodes", type=int, default=20)
    evaluating.add_argument("--quantiles", type=int, default=64)
    evaluating.add_argument("--risk-weight", type=float, help="override training tradeoff without retraining")
    evaluating.add_argument("--video")
    evaluating.set_defaults(func=evaluate)

    info = commands.add_parser("model-info", parents=[common_model])
    info.set_defaults(func=model_info)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
