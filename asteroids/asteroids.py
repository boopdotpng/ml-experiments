"""A compact, from-scratch PPO agent for headless Atari Asteroids.

Gymnasium/ALE supplies pixels and game dynamics.  Everything agent-side--the
network, rollout storage, GAE, PPO loss, training loop, and checkpoints--lives
in this file so the whole algorithm is easy to inspect.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any

import ale_py
import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics
from torch import nn
from torch.distributions.categorical import Categorical


ENV_ID = "ALE/Asteroids-v5"
FRAME_SKIP = 4
STACK_SIZE = 4
gym.register_envs(ale_py)
# AsyncVectorEnv already parallelizes across processes. Letting each OpenCV
# resize create a 12-thread pool produced hundreds of native threads and
# intermittent worker segfaults on this 12-thread host.
cv2.setNumThreads(1)


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
    """Create one sticky-action Asteroids environment with standard preprocessing."""
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=FRAME_SKIP,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, STACK_SIZE)
    env = RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """The classic Atari visual encoder with separate policy and value heads."""

    def __init__(self, actions: int = 14) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(STACK_SIZE, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # A small policy initialization prevents a strongly biased initial action.
        self.policy = layer_init(nn.Linear(512, actions), std=0.01)
        self.value = layer_init(nn.Linear(512, 1), std=1.0)

    def hidden(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs.float().div_(255.0))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(self.hidden(obs)).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.hidden(obs)
        distribution = Categorical(logits=self.policy(hidden))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.value(hidden).squeeze(-1)


def signed_sqrt(reward: np.ndarray) -> np.ndarray:
    """Compress Atari score magnitudes while preserving their ordering and sign."""
    return np.sign(reward) * np.sqrt(np.abs(reward))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over [time, environment] arrays."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        following_value = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        alive = 1.0 - dones[step]
        delta = rewards[step] + gamma * following_value * alive - values[step]
        last_advantage = delta + gamma * gae_lambda * alive * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def save_checkpoint(
    path: Path,
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    frames: int,
    update: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "frames": frames,
            "update": update,
            "args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_agent(checkpoint: Path, device: torch.device) -> tuple[Agent, dict[str, Any]]:
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = Agent().to(device)
    agent.load_state_dict(data["model"])
    agent.eval()
    return agent, data


def episode_stats(infos: dict[str, Any]) -> list[tuple[float, int]]:
    """Extract completed episode (raw score, agent steps) from same-step autoreset info."""
    final = infos.get("final_info")
    if not isinstance(final, dict) or "episode" not in final:
        return []
    episode = final["episode"]
    valid = final.get("_episode", np.ones_like(episode["r"], dtype=bool))
    return [(float(r), int(length)) for r, length, keep in zip(episode["r"], episode["l"], valid) if keep]


def final_lives(infos: dict[str, Any], done: np.ndarray) -> np.ndarray:
    """Return lives before same-step autoreset, allowing correct life-loss shaping."""
    lives = np.asarray(infos["lives"]).copy()
    final = infos.get("final_info")
    if isinstance(final, dict) and "lives" in final:
        mask = np.asarray(infos.get("_final_info", done), dtype=bool)
        lives[mask] = np.asarray(final["lives"])[mask]
    return lives


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device == "cuda":
        raise SystemExit("CUDA was requested but is unavailable; pass --device cpu to run anyway")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # A top-level callable plus plain arguments stays picklable under forkserver.
    env_fns = [partial(make_env, args.seed + i) for i in range(args.num_envs)]
    envs = gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=True,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    agent = Agent(envs.single_action_space.n).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    frames = 0
    first_update = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        frames = int(checkpoint.get("frames", 0))
        first_update = int(checkpoint.get("update", 0)) + 1
        print(f"resumed {args.resume} at {frames:,} emulator frames")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.csv"
    log_exists = log_path.exists() and args.resume
    log_file = log_path.open("a" if log_exists else "w", newline="")
    fields = ["update", "frames", "fps", "score_100", "length_100", "policy_loss", "value_loss", "entropy", "approx_kl", "learning_rate"]
    logger = csv.DictWriter(log_file, fieldnames=fields)
    if not log_exists:
        logger.writeheader()

    obs_np, infos = envs.reset(seed=[args.seed + i for i in range(args.num_envs)])
    obs = torch.as_tensor(obs_np, dtype=torch.uint8, device=device)
    last_lives = np.asarray(infos["lives"])
    scores: deque[float] = deque(maxlen=100)
    lengths: deque[int] = deque(maxlen=100)
    start_time = time.monotonic()
    start_frames = frames
    rollout_frames = args.num_envs * args.rollout_steps * FRAME_SKIP
    total_updates = math.ceil(max(0, args.total_frames - frames) / rollout_frames)

    # Rollouts remain uint8 until a sampled minibatch reaches the network.
    rollout_obs = torch.empty((args.rollout_steps, args.num_envs, STACK_SIZE, 84, 84), dtype=torch.uint8, device=device)
    rollout_actions = torch.empty((args.rollout_steps, args.num_envs), dtype=torch.long, device=device)
    rollout_logprobs = torch.empty((args.rollout_steps, args.num_envs), device=device)
    rollout_rewards = torch.empty((args.rollout_steps, args.num_envs), device=device)
    rollout_dones = torch.empty((args.rollout_steps, args.num_envs), device=device)
    rollout_values = torch.empty((args.rollout_steps, args.num_envs), device=device)

    print(f"training on {device}: {args.num_envs} actors, {args.total_frames:,} emulator frames")
    try:
        for update_offset in range(total_updates):
            update = first_update + update_offset
            progress = min(1.0, frames / max(1, args.total_frames))
            learning_rate = args.learning_rate * (1.0 - progress) if args.anneal_lr else args.learning_rate
            optimizer.param_groups[0]["lr"] = learning_rate

            agent.eval()
            for step in range(args.rollout_steps):
                rollout_obs[step].copy_(obs)
                with torch.inference_mode():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                rollout_actions[step] = action
                rollout_logprobs[step] = logprob
                rollout_values[step] = value

                next_obs_np, raw_reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done_np = np.logical_or(terminated, truncated)
                before_reset_lives = final_lives(infos, done_np)
                lost_life = before_reset_lives < last_lives
                shaped_reward = signed_sqrt(raw_reward) - args.life_penalty * lost_life
                rollout_rewards[step] = torch.as_tensor(shaped_reward, device=device)
                rollout_dones[step] = torch.as_tensor(done_np, dtype=torch.float32, device=device)

                for score, length in episode_stats(infos):
                    scores.append(score)
                    lengths.append(length * FRAME_SKIP)
                obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)
                last_lives = np.asarray(infos["lives"])
                frames += args.num_envs * FRAME_SKIP

            with torch.inference_mode():
                next_value = agent.get_value(obs)
                advantages, returns = compute_gae(
                    rollout_rewards, rollout_values, rollout_dones, next_value, args.gamma, args.gae_lambda
                )

            batch_obs = rollout_obs.flatten(0, 1)
            batch_actions = rollout_actions.flatten()
            batch_logprobs = rollout_logprobs.flatten()
            batch_advantages = advantages.flatten()
            batch_returns = returns.flatten()
            batch_values = rollout_values.flatten()
            batch_size = batch_actions.numel()
            indices = np.arange(batch_size)
            metrics: list[tuple[float, float, float, float]] = []

            agent.train()
            stop_early = False
            for _ in range(args.epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, args.minibatch_size):
                    mb = torch.as_tensor(indices[start : start + args.minibatch_size], device=device)
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(batch_obs[mb], batch_actions[mb])
                    log_ratio = new_logprob - batch_logprobs[mb]
                    ratio = log_ratio.exp()
                    advantage = batch_advantages[mb]
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    policy_loss = torch.maximum(
                        -advantage * ratio,
                        -advantage * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef),
                    ).mean()

                    unclipped_value_loss = (new_value - batch_returns[mb]).square()
                    clipped_value = batch_values[mb] + (new_value - batch_values[mb]).clamp(-args.clip_coef, args.clip_coef)
                    value_loss = 0.5 * torch.maximum(unclipped_value_loss, (clipped_value - batch_returns[mb]).square()).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    metrics.append((policy_loss.item(), value_loss.item(), entropy_loss.item(), approx_kl.item()))
                    if args.target_kl and approx_kl > args.target_kl:
                        stop_early = True
                        break
                if stop_early:
                    break

            elapsed = time.monotonic() - start_time
            fps = int((frames - start_frames) / max(elapsed, 1e-6))
            mean_metrics = np.mean(metrics, axis=0)
            row = {
                "update": update,
                "frames": frames,
                "fps": fps,
                "score_100": round(float(np.mean(scores)), 2) if scores else "",
                "length_100": round(float(np.mean(lengths)), 1) if lengths else "",
                "policy_loss": float(mean_metrics[0]),
                "value_loss": float(mean_metrics[1]),
                "entropy": float(mean_metrics[2]),
                "approx_kl": float(mean_metrics[3]),
                "learning_rate": learning_rate,
            }
            logger.writerow(row)
            log_file.flush()
            if update == first_update or update % args.log_every == 0:
                score_text = f"{np.mean(scores):.1f}" if scores else "waiting"
                print(f"frames={frames:>10,}  fps={fps:>5}  score100={score_text:>8}  entropy={mean_metrics[2]:.3f}  kl={mean_metrics[3]:.4f}")
            if update % args.save_every == 0 or frames >= args.total_frames:
                save_checkpoint(output / "latest.pt", agent, optimizer, frames, update, args)
                if update % (args.save_every * 10) == 0 or frames >= args.total_frames:
                    save_checkpoint(output / f"frames-{frames}.pt", agent, optimizer, frames, update, args)
    except KeyboardInterrupt:
        print("interrupted; saving current policy")
        save_checkpoint(output / "latest.pt", agent, optimizer, frames, update, args)
    finally:
        envs.close()
        log_file.close()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    agent, data = load_agent(Path(args.checkpoint), device)
    env = make_env(args.seed, render_mode="rgb_array" if args.video else None)
    writer = None
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.video, fps=15, codec="libx264", quality=8, macro_block_size=2)
    scores: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            score = 0.0
            steps = 0
            done = False
            while not done:
                tensor = torch.as_tensor(obs, dtype=torch.uint8, device=device).unsqueeze(0)
                hidden = agent.hidden(tensor)
                logits = agent.policy(hidden)
                action = Categorical(logits=logits).sample().item() if args.sample_actions else logits.argmax(dim=-1).item()
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
    print(
        f"checkpoint frames: {int(data.get('frames', 0)):,}\n"
        f"score  mean={np.mean(scores):.1f}  median={np.median(scores):.1f}  min={np.min(scores):.1f}  max={np.max(scores):.1f}\n"
        f"length mean={np.mean(lengths):,.0f} emulator frames"
    )


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    common.add_argument("--seed", type=int, default=1)
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    training = commands.add_parser("train", parents=[common], help="train a policy")
    training.add_argument("--total-frames", type=int, default=50_000_000, help="emulator frames, including action repeats")
    training.add_argument("--num-envs", type=int, default=8)
    training.add_argument("--rollout-steps", type=int, default=256)
    training.add_argument("--minibatch-size", type=int, default=512)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=2.5e-4)
    training.add_argument("--gamma", type=float, default=0.997)
    training.add_argument("--gae-lambda", type=float, default=0.95)
    training.add_argument("--clip-coef", type=float, default=0.1)
    training.add_argument("--entropy-coef", type=float, default=0.01)
    training.add_argument("--value-coef", type=float, default=0.5)
    training.add_argument("--max-grad-norm", type=float, default=0.5)
    training.add_argument("--target-kl", type=float, default=0.03, help="0 disables PPO early stopping")
    training.add_argument("--life-penalty", type=float, default=5.0)
    training.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--output", default="asteroids/runs/default")
    training.add_argument("--resume", type=Path)
    training.add_argument("--log-every", type=int, default=10)
    training.add_argument("--save-every", type=int, default=100)
    training.set_defaults(func=train)

    evaluating = commands.add_parser("eval", parents=[common], help="evaluate a checkpoint")
    evaluating.add_argument("checkpoint", type=Path)
    evaluating.add_argument("--episodes", type=int, default=20)
    evaluating.add_argument("--video", help="record the first episode to this MP4")
    evaluating.add_argument("--sample-actions", action="store_true", help="sample the PPO policy instead of taking its mode")
    evaluating.set_defaults(func=evaluate)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
