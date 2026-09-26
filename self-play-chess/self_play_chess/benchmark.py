"""Read-only checkpoint evaluation. Stockfish data never enters training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.request import Request, urlopen

import chess
import chess.engine
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import load_state_dict, safe_load, safe_load_metadata

from .encoding import encode
from .environment import CLAIM_DRAW, ChessEnv, move_to_action
from .model import ChessTransformer, ModelConfig, legal_policy
from .inference import Inference


def evaluate_game(model, color, engine, profile, seed, deadline, max_plies=256):
    env, rng = ChessEnv(), np.random.default_rng(seed)
    token = object()  # New game resets the engine's state/hash via UCI.
    for ply in range(max_plies):
        if time.time() >= deadline:
            return {"result": "*", "reason": "TIME_BUDGET", "plies": ply, "score": None}
        if env.outcome() is not None:
            break
        if env.board.turn == color:
            observation = encode([env])
            probabilities, _ = model(observation)
            policy = probabilities[0].astype(np.float64)
            policy /= policy.sum()
            env.step(int(rng.choice(len(policy), p=policy)))
        elif profile["name"] == "random":
            choices = np.flatnonzero(env.legal_mask())
            env.step(int(rng.choice(choices)))
        elif env.board.can_claim_draw():
            env.step(CLAIM_DRAW)
        else:
            result = engine.play(env.board, chess.engine.Limit(**profile["limit"]), game=token)
            env.step(move_to_action(result.move))
    outcome = env.outcome()
    return {"result": outcome.result() if outcome else "*", "reason": outcome.termination.name if outcome else "PLY_LIMIT",
            "plies": len(env.board.move_stack), "score": None if outcome is None else
            (0.5 if outcome.winner is None else float(outcome.winner == color))}


def summarize(games):
    output = []
    for name, opponent in dict.fromkeys((game["model"], game["opponent"]) for game in games):
        rows = [game for game in games if game["model"] == name and game["opponent"] == opponent]
        wins = sum(game["score"] == 1 for game in rows)
        draws = sum(game["score"] == 0.5 for game in rows)
        losses = sum(game["score"] == 0 for game in rows)
        unresolved = len(rows) - wins - draws - losses
        output.append(dict(model=name, opponent=opponent, wins=wins, draws=draws, losses=losses,
                           unresolved=unresolved, games=len(rows),
                           score=None if wins + draws + losses == 0 else (wins + draws / 2) / (wins + draws + losses)))
    return output


def benchmark(run_dir, stockfish, deadline, initial=False):
    snapshot = run_dir / "evaluation-snapshot.safetensors"
    shutil.copyfile(run_dir / "latest.safetensors", snapshot)
    metadata = json.loads(safe_load_metadata(snapshot)[2]["__metadata__"]["run"])
    Tensor.manual_seed(metadata["training_config"]["seed"])
    models = {name: ChessTransformer(ModelConfig(**metadata["model_config"])) for name in ("A", "B")}
    if not initial:
        weights = {key: value for key, value in safe_load(snapshot).items() if key.startswith("models.")}
        load_state_dict({"models": models}, weights, verbose=False)
    models = {name: Inference(model) for name, model in models.items()}
    games = []
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish.resolve()), timeout=20)
    engine_id = engine.id
    try:
        elo_min = engine.options["UCI_Elo"].min
        profiles = [
            {"name": "random", "options": {}, "limit": {}},
            {"name": "depth 1", "options": {"Skill Level": 20, "UCI_LimitStrength": False}, "limit": {"depth": 1}},
            {"name": "100 nodes", "options": {"Skill Level": 20, "UCI_LimitStrength": False}, "limit": {"nodes": 100}},
            {"name": f"requested Elo {elo_min} / 10ms", "options": {"UCI_LimitStrength": True, "UCI_Elo": elo_min}, "limit": {"time": 0.01}},
        ]
        for profile in profiles:
            engine.configure({"Threads": 1, "Hash": 16, **profile["options"]})
            for name in ("A", "B"):
                for color in (chess.WHITE, chess.BLACK):
                    if time.time() >= deadline:
                        break
                    game = evaluate_game(models[name], color, engine, profile, 2026 + int(color), deadline)
                    game.update(model=name, opponent=profile["name"], color="white" if color else "black")
                    games.append(game)
    finally:
        engine.quit()
    report = {"time": datetime.now(timezone.utc).isoformat(), "checkpoint_games": 0 if initial else metadata["games"],
              "checkpoint_updates": 0 if initial else metadata["updates"], "initial": initial,
              "engine": engine_id, "profiles": profiles, "games": games, "summary": summarize(games),
              "note": "Two games per model/opponent is a smoke benchmark, not an Elo estimate. Unfinished games excluded from score; reported separately. Requested engine Elo at 10ms is not calibrated player Elo."}
    with (run_dir / "benchmarks.jsonl").open("a") as file:
        file.write(json.dumps(report) + "\n")
    temporary = run_dir / "evaluation.tmp.json"
    temporary.write_text(json.dumps(report))
    os.replace(temporary, run_dir / "evaluation.json")
    print(json.dumps({"checkpoint_games": report["checkpoint_games"], "initial": initial, "summary": report["summary"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    if args.minutes <= 0 or args.interval <= 0:
        parser.error("minutes and interval must be positive")
    deadline = time.time() + args.minutes * 60
    schedule = {"deadline": deadline, "started": time.time(), "minutes": args.minutes}
    (args.run_dir / "monitor.json").write_text(json.dumps(schedule))
    stopped = threading.Event()

    def stop_training():
        # Wall-clock stop is independent of how long evaluation takes.
        time.sleep(max(0, deadline - time.time()))
        for attempt in range(5):
            try:
                with urlopen(Request(args.url + "/api/stop", method="POST",
                         headers={"Origin": args.url, "X-Chess-Control": "1"}), timeout=5):
                    pass
                stopped.set()
                return
            except OSError as error:
                print(f"Stop request attempt {attempt + 1}: {error}", flush=True)
                time.sleep(2)
        (args.run_dir / "monitor-error.txt").write_text("Could not stop trainer automatically; stop it in the browser.")

    stopper = threading.Thread(target=stop_training, daemon=False)
    stopper.start()
    print(f"Monitoring until {datetime.fromtimestamp(deadline).astimezone().isoformat()}", flush=True)
    try:
        benchmark(args.run_dir, args.stockfish, min(deadline, time.time() + 240), initial=True)
        while time.time() < deadline:
            benchmark(args.run_dir, args.stockfish, min(deadline, time.time() + 240))
            stopped.wait(min(args.interval, max(0, deadline - time.time())))
    finally:
        stopper.join()
    # Give an in-flight optimizer update time to save, then evaluate frozen final weights.
    for _ in range(120):
        with urlopen(args.url + "/api/state", timeout=5) as response:
            if json.load(response)["phase"] in ("stopped", "error"):
                break
        time.sleep(1)
    benchmark(args.run_dir, args.stockfish, time.time() + 240)
    (args.run_dir / "monitor-complete.txt").write_text("Timed training ended; final benchmark saved.\n")


if __name__ == "__main__":
    main()
