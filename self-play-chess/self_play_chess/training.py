"""On-policy Monte Carlo actor-critic: two learners, terminal rewards only."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
import time

import chess
import chess.pgn
import chess.svg
import numpy as np
from tinygrad import Context, Device, Tensor, nn
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_load_metadata, safe_save

from .encoding import Observation, encode
from .environment import CLAIM_DRAW, ChessEnv, action_to_move
from .model import ChessTransformer, ModelConfig, legal_policy
from .inference import Inference


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.0003
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    batch_size: int = 8
    max_plies: int = 0  # 0 means play until a rule-defined outcome
    move_delay: float = 0.1
    seed: int = 42
    parallel_games: int = 1
    updates_per_round: int = 1
    policy_clip: float = 0.2
    outcome_stratified: bool = False

    def __post_init__(self):
        if self.learning_rate <= 0 or self.batch_size < 1 or self.max_plies < 0 or self.move_delay < 0 or self.parallel_games < 1:
            raise ValueError("Invalid training configuration")
        if self.entropy_weight < 0 or self.value_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        if self.updates_per_round < 1 or not 0 < self.policy_clip < 1:
            raise ValueError("Invalid update count or policy clipping range")


class LiveState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pause = threading.Event()
        self.stop = threading.Event()
        self.data = {"phase": "starting", "game": 1, "ply": 0, "moves": [], "history": [],
                     "stats": {"A": 0, "B": 0, "draws": 0, "truncated": 0}, "metrics": {},
                     "board_svg": chess.svg.board(chess.Board()), "message": "Initializing two models…"}

    def publish(self, **fields):
        with self.lock:
            self.data.update(fields)

    def snapshot(self):
        with self.lock:
            return dict(self.data, paused=self.pause.is_set(), stopping=self.stop.is_set())

    def ready(self):
        while self.pause.is_set() and not self.stop.is_set():
            self.stop.wait(0.1)
        return not self.stop.is_set()


def outcome_target(outcome: chess.Outcome, color: chess.Color) -> tuple[float, int]:
    if outcome.winner is None:
        return 0.0, 1
    return (1.0, 2) if outcome.winner == color else (-1.0, 0)


def actor_critic_loss(logits, value_logits, legal, actions, returns, classes,
                      entropy_weight=0.01, value_weight=0.5, old_log_probs=None, advantages=None, clip=0.2,
                      sample_weights=None):
    # Finite masked log probabilities avoid 0 * -inf in entropy/backprop.
    log_probs = legal.where(logits, -1e9).log_softmax(axis=-1)
    probabilities = log_probs.exp()
    wdl = value_logits.softmax(axis=-1)
    advantage = (returns - (wdl[:, 2] - wdl[:, 0])).detach() if advantages is None else advantages.detach()
    selected = log_probs.gather(1, actions.reshape(-1, 1)).reshape(-1)
    weights = Tensor.ones(*returns.shape) if sample_weights is None else sample_weights
    if old_log_probs is None:
        policy = -(weights * advantage * selected).mean()
    else:
        ratio = (selected - old_log_probs.detach()).exp()
        policy = -(weights * (ratio * advantage).minimum(ratio.clip(1-clip, 1+clip) * advantage)).mean()
    value = -(weights * value_logits.log_softmax(axis=-1).gather(1, classes.reshape(-1, 1)).reshape(-1)).mean()
    entropy = -(weights * (probabilities * log_probs).sum(axis=-1)).mean()
    return policy + value_weight * value - entropy_weight * entropy, policy, value, entropy


def pack_observations(rows: list[dict[str, np.ndarray]]) -> Observation:
    return Observation(**{name: Tensor(np.concatenate([row[name] for row in rows], axis=0))
                          for name in Observation.__dataclass_fields__})


def sample_training_indices(classes, batch_size, steps, rng, stratified=False):
    """Ensure outcome coverage, correcting back to the uniform position objective."""
    classes = np.asarray(classes)
    count = batch_size * steps
    groups = [np.flatnonzero(classes == label) for label in np.unique(classes)]
    if not stratified or batch_size < len(groups):
        return rng.choice(len(classes), count, replace=len(classes) < count), np.ones(count, dtype=np.float32)
    indices, weights = [], []
    for _ in range(steps):
        allocation = np.full(len(groups), batch_size // len(groups))
        allocation[rng.permutation(len(groups))[:batch_size % len(groups)]] += 1
        batch_indices, batch_weights = [], []
        for group, size in zip(groups, allocation):
            batch_indices.extend(rng.choice(group, size, replace=len(group) < size))
            # P(outcome in trajectory) / P(outcome in this minibatch).
            batch_weights.extend([len(group) / len(classes) * batch_size / size] * size)
        order = rng.permutation(batch_size)
        indices.extend(np.asarray(batch_indices)[order])
        weights.extend(np.asarray(batch_weights)[order])
    return np.asarray(indices), np.asarray(weights, dtype=np.float32)


class Trainer:
    def __init__(self, live: LiveState, run_dir: Path, config: TrainingConfig = TrainingConfig(),
                 model_config: ModelConfig = ModelConfig(width=32, layers=1, policy_width=8), resume=False):
        self.live, self.run_dir = live, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.run_dir / "latest.safetensors"
        metadata = None
        if resume:
            metadata = json.loads(safe_load_metadata(self.checkpoint)[2]["__metadata__"]["run"])
            config = TrainingConfig(**metadata["training_config"])
            model_config = ModelConfig(**metadata["model_config"])
        elif self.checkpoint.exists():
            raise ValueError("Run directory already has a checkpoint; use --resume or a new --run-dir")
        self.config, self.model_config = config, model_config
        Tensor.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.models = {name: ChessTransformer(model_config) for name in ("A", "B")}
        self.optimizers = {name: nn.optim.Adam(get_parameters(model), lr=config.learning_rate)
                           for name, model in self.models.items()}
        self.games, self.updates = 0, 0
        self.training_samples = 0
        self.stats = {"A": 0, "B": 0, "draws": 0, "truncated": 0}
        if metadata:
            load_state_dict(self.state_objects(), safe_load(self.checkpoint), verbose=False)
            self.games, self.updates, self.stats = metadata["games"], metadata["updates"], metadata["stats"]
            self.training_samples = metadata.get("training_samples", self.updates * config.batch_size)
            self.rng.bit_generator.state = metadata["rng"]
        self.started = time.monotonic()
        self.inference = {name: Inference(model) for name, model in self.models.items()}
        self.live.publish(device=Device.DEFAULT, parameters=sum(p.numel() for p in get_parameters(self.models["A"])),
                          model_config=asdict(model_config), training_config=asdict(config),
                          run_dir=str(self.run_dir.resolve()), updates=self.updates,
                          training_samples=self.training_samples, stats=dict(self.stats))

    def state_objects(self):
        # Only optimizer buffers; model parameters are saved once.
        optimizer_buffers = {name: {key: getattr(opt, key) for key in ("m", "v", "b1_t", "b2_t", "lr")}
                             for name, opt in self.optimizers.items()}
        return {"models": self.models, "optimizers": optimizer_buffers}

    def save(self):
        metadata = {"model_config": asdict(self.model_config), "training_config": asdict(self.config),
                    "games": self.games, "updates": self.updates, "stats": self.stats, "training_samples": self.training_samples,
                    "rng": self.rng.bit_generator.state}
        temporary = self.checkpoint.with_suffix(".tmp")
        safe_save(get_state_dict(self.state_objects()), str(temporary), {"run": json.dumps(metadata)})
        os.replace(temporary, self.checkpoint)

    def update(self, name, trajectory, outcome):
        return self.update_samples(name, [(row, action, *outcome_target(outcome, color)) for row, action, color in trajectory])

    def update_samples(self, name, trajectory):
        if not trajectory:
            return {}
        count = self.config.batch_size * self.config.updates_per_round
        indices, weights = sample_training_indices([sample[3] for sample in trajectory],
            self.config.batch_size, self.config.updates_per_round, self.rng, self.config.outcome_stratified)
        # Freeze behavior log probabilities AND advantages before any optimizer step.
        # The model has not changed since collecting this learner's rollouts.
        batches = []
        with Context(TRAINING=1, BEAM=getenv("TRAIN_BEAM", 0)):
            for offset in range(0, count, self.config.batch_size):
                samples = [trajectory[i] for i in indices[offset:offset+self.config.batch_size]]
                observation = pack_observations([sample[0] for sample in samples])
                actions = Tensor(np.array([sample[1] for sample in samples], dtype=np.int32))
                returns = Tensor(np.array([sample[2] for sample in samples], dtype=np.float32))
                classes = Tensor(np.array([sample[3] for sample in samples], dtype=np.int32))
                logits, values = self.models[name](observation)
                logp = observation.legal.where(logits, -1e9).log_softmax(axis=-1).gather(1, actions.reshape(-1, 1)).reshape(-1)
                wdl = values.softmax(axis=-1)
                old_logp = Tensor(logp.numpy().copy())
                advantage = Tensor((returns - (wdl[:, 2] - wdl[:, 0])).numpy().copy())
                sample_weights = Tensor(weights[offset:offset+self.config.batch_size])
                batches.append((observation, actions, returns, classes, old_logp, advantage, sample_weights))
            metrics, norms = [], []
            for step, (observation, actions, returns, classes, old_logp, advantage, sample_weights) in enumerate(batches):
                self.live.publish(message=f"Model {name}: learning step {step+1}/{len(batches)} · {count} sampled positions")
                logits, values = self.models[name](observation)
                losses = actor_critic_loss(logits, values, observation.legal, actions, returns, classes,
                                          self.config.entropy_weight, self.config.value_weight,
                                          old_logp, advantage, self.config.policy_clip, sample_weights)
                measured = [float(loss.item()) for loss in losses]
                if not np.isfinite(measured).all():
                    raise FloatingPointError("Non-finite training loss")
                optimizer = self.optimizers[name]
                optimizer.zero_grad()
                losses[0].backward()
                norm = sum((p.grad.square().sum() for p in optimizer.params)).sqrt()
                norm_value = float(norm.item())
                if not np.isfinite(norm_value):
                    raise FloatingPointError("Non-finite gradient")
                for parameter in optimizer.params:
                    parameter.grad = parameter.grad / max(1.0, norm_value)
                optimizer.step()
                metrics.append(measured)
                norms.append(norm_value)
                self.updates += 1
                self.training_samples += self.config.batch_size
                self.live.publish(updates=self.updates, training_samples=self.training_samples)
        self.inference[name] = Inference(self.models[name])
        return dict(zip(("loss", "policy_loss", "value_loss", "entropy"), np.mean(metrics, axis=0).tolist()),
                    gradient_norm=float(np.mean(norms)), samples=count, optimizer_steps=len(batches),
                    available_positions=len(trajectory),
                    decisive_samples=sum(trajectory[i][3] != 1 for i in indices))

    def play_game(self):
        env, trajectories, moves = ChessEnv(), {"A": [], "B": []}, []
        white = "A" if self.games % 2 == 0 else "B"
        black = "B" if white == "A" else "A"
        game_started = time.monotonic()
        self.live.publish(game=self.games + 1, ply=0, moves=[], white=white, black=black,
                          board_svg=chess.svg.board(env.board), fen=env.board.fen(), phase="playing",
                          message="Models sample legal moves. No engine or human guidance.", predictions=None)
        while env.outcome() is None:
            if not self.live.ready():
                return False
            if self.config.max_plies and len(moves) >= self.config.max_plies:
                break
            color = env.board.turn
            name = white if color else black
            started = time.monotonic()
            observation = encode([env])
            logits, values = self.models[name](observation)
            policy = legal_policy(logits, observation.legal).numpy()[0].astype(np.float64)
            policy /= policy.sum()
            predictions = values.softmax(axis=-1).numpy()[0].tolist()
            action = int(self.rng.choice(len(policy), p=policy))
            arrays = {field: getattr(observation, field).numpy().copy() for field in Observation.__dataclass_fields__}
            trajectories[name].append((arrays, action, color))
            move = None if action == CLAIM_DRAW else action_to_move(action)
            san = "claim draw" if move is None else env.board.san(move)
            env.step(action)
            moves.append(san)
            elapsed = time.monotonic() - started
            self.live.publish(phase="playing", ply=len(moves), moves=list(moves), turn="White" if env.board.turn else "Black",
                              board_svg=chess.svg.board(env.board, lastmove=move,
                                  check=env.board.king(env.board.turn) if env.board.is_check() else None,
                                  colors={"square light": "#eadfce", "square dark": "#8faaa0"}),
                              fen=env.board.fen(), last_move=san, move_seconds=elapsed,
                              game_seconds=time.monotonic() - game_started, predictions=predictions,
                              prediction_player=name, elapsed_seconds=time.monotonic() - self.started)
            if self.live.stop.wait(max(0, self.config.move_delay - elapsed)):
                return False
        outcome = env.outcome()
        metrics = {}
        if outcome is not None:
            self.live.publish(phase="training", message=f"{outcome.result()} · {outcome.termination.name}. Updating both models…")
            for name in ("A", "B"):
                metrics[name] = self.update(name, trajectories[name], outcome)
            self.stats["draws" if outcome.winner is None else (white if outcome.winner else black)] += 1
        else:
            self.stats["truncated"] += 1  # Never call an unfinished game a draw.
        self.games += 1
        result = {"game": self.games, "result": outcome.result() if outcome else "*",
                  "reason": outcome.termination.name if outcome else "PLY_LIMIT_NO_TRAINING",
                  "plies": len(moves), "white": white, "black": black, "metrics": metrics}
        game = chess.pgn.Game.from_board(env.board)
        game.headers.update(White=f"Model {white}", Black=f"Model {black}", Result=result["result"],
                            Event="tinygrad self-play", Termination=result["reason"])
        with (self.run_dir / "games.pgn").open("a") as file:
            file.write(str(game) + "\n\n")
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(result) + "\n")
        self.save()
        history = [result] + self.live.snapshot()["history"][:19]
        self.live.publish(phase="between games", history=history, stats=dict(self.stats), metrics=metrics,
                          updates=self.updates, message=f"Game {self.games}: {result['result']} · checkpoint saved")
        print(json.dumps(result), flush=True)
        self.live.stop.wait(2)
        return True

    def run(self, games=0):
        completed = 0
        self.save()  # Keep a recoverable starting checkpoint even before game one.
        while (games == 0 or completed < games) and self.live.ready():
            count = min(self.config.parallel_games, games - completed) if games else self.config.parallel_games
            if not (self.play_parallel(count) if count > 1 else self.play_game()):
                break
            completed += count
        self.save()
        self.live.publish(phase="stopped", message="Checkpoint saved. Unfinished game, if any, was not trained on.")

    def play_parallel(self, count, learn=True):
        """Freeze both policies for a wave, batch boards by actor, then update once each."""
        envs = [ChessEnv() for _ in range(count)]
        moves, trajectories = [[] for _ in envs], [[] for _ in envs]
        white = ["A" if (self.games + i) % 2 == 0 else "B" for i in range(count)]
        done = set()
        base_game = self.games
        batch_size = (count + 1) // 2
        start = time.monotonic()
        total_plies = 0
        self.live.publish(phase="playing", parallel_games=count, active_games=count,
                          message=f"{count} parallel games · batched TinyJit inference · fixed weights during this wave")
        while len(done) < count and self.live.ready():
            tick = time.monotonic()
            # Choose both groups before mutating boards, so each game advances one ply.
            groups = {actor: [i for i, env in enumerate(envs) if i not in done and
                      (white[i] if env.board.turn else ("B" if white[i] == "A" else "A")) == actor]
                      for actor in ("A", "B")}
            for name in ("A", "B"):
                indices = groups[name]
                if not indices:
                    continue
                padded = indices + [indices[-1]] * (batch_size - len(indices))
                observation = encode([envs[i] for i in padded])
                policies, values = self.inference[name](observation)
                arrays = {field: getattr(observation, field).numpy() for field in Observation.__dataclass_fields__}
                for row, i in enumerate(indices):
                    env = envs[i]
                    policy = policies[row].astype(np.float64)
                    policy /= policy.sum()
                    action = int(self.rng.choice(len(policy), p=policy))
                    saved = {field: array[row:row+1].copy() for field, array in arrays.items()}
                    trajectories[i].append((name, saved, action, env.board.turn))
                    move = None if action == CLAIM_DRAW else action_to_move(action)
                    moves[i].append("claim draw" if move is None else env.board.san(move))
                    env.step(action)
                    total_plies += 1
                    if env.outcome() is not None or (self.config.max_plies and len(moves[i]) >= self.config.max_plies):
                        done.add(i)
                    # Follow the first unfinished board; show its actual game number/color pairing.
                    if i == next((j for j in range(count) if j not in done), i if len(done) == count else -1):
                        self.live.publish(game=base_game+i+1, ply=len(moves[i]), moves=list(moves[i]), white=white[i],
                            black="B" if white[i] == "A" else "A", fen=env.board.fen(),
                            board_svg=chess.svg.board(env.board, lastmove=move), predictions=values[row].tolist(),
                            prediction_player=name, active_games=count-len(done),
                            positions_per_second=total_plies / max(0.001, time.monotonic()-start))
            self.live.publish(move_seconds=(time.monotonic()-tick)/max(1, count-len(done)))
            self.live.stop.wait(max(0, self.config.move_delay-(time.monotonic()-tick)))
        samples = {"A": [], "B": []}
        results = []
        # Stopping keeps terminal games and discards only unfinished trajectories.
        for i, env in enumerate(envs):
            outcome = env.outcome()
            if outcome is None and i not in done:
                continue
            if outcome is not None:
                for name, row, action, color in trajectories[i]:
                    samples[name].append((row, action, *outcome_target(outcome, color)))
                self.stats["draws" if outcome.winner is None else (white[i] if outcome.winner else ("B" if white[i] == "A" else "A"))] += 1
            else:
                self.stats["truncated"] += 1
            self.games += 1
            result = dict(game=self.games, result=outcome.result() if outcome else "*", plies=len(moves[i]),
                          reason=outcome.termination.name if outcome else "PLY_LIMIT_NO_TRAINING", white=white[i],
                          black="B" if white[i] == "A" else "A")
            results.append(result)
            game = chess.pgn.Game.from_board(env.board)
            game.headers.update(White=f"Model {result['white']}", Black=f"Model {result['black']}", Result=result["result"])
            with (self.run_dir / "games.pgn").open("a") as file:
                file.write(str(game) + "\n\n")
        self.live.publish(phase="training", message=f"Updating from {len(results)} completed games in this wave…")
        metrics = {name: self.update_samples(name, samples[name]) for name in ("A", "B") if samples[name]} if learn else {}
        for result in results:
            result["metrics"] = metrics
            result["wave_updates"] = self.updates
            with (self.run_dir / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(result) + "\n")
        self.save()
        self.live.publish(updates=self.updates, metrics=metrics, stats=dict(self.stats),
                          history=(list(reversed(results))+self.live.snapshot()["history"])[:20])
        print(json.dumps({"games": self.games, "updates": self.updates, "stats": self.stats,
                          "wave_seconds": time.monotonic()-start, "metrics": metrics}), flush=True)
        return not self.live.stop.is_set()
