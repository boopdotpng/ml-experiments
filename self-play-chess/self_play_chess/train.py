"""Local trainer and live viewer. Run: python -m self_play_chess.train"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback

from .model import ModelConfig
from .training import LiveState, Trainer, TrainingConfig


def make_server(live, port):
    page = (Path(__file__).parent / "static" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, data, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                state = live.snapshot()
                if state.get("run_dir"):
                    for key, filename in (("evaluation", "evaluation.json"), ("monitor", "monitor.json")):
                        try:
                            state[key] = json.loads((Path(state["run_dir"]) / filename).read_text())
                        except (OSError, ValueError):
                            pass
                self.respond(200, json.dumps(state).encode())
            else:
                self.respond(404, b'{}')

        def do_POST(self):
            # Local UI only: reject cross-site control requests.
            expected_origin = f"http://{self.headers.get('Host')}"
            if self.headers.get("Origin") != expected_origin or self.headers.get("X-Chess-Control") != "1":
                self.respond(403, b'{}')
                return
            if self.path == "/api/pause":
                live.pause.set()
            elif self.path == "/api/resume":
                live.pause.clear()
            elif self.path == "/api/stop":
                live.stop.set()
            else:
                self.respond(404, b'{}')
                return
            self.respond(200, b'{"ok":true}')

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--games", type=int, default=0, help="0: keep playing until stopped")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--policy-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None, help="Training minibatch size; overrides saved setting")
    parser.add_argument("--outcome-stratified", action=argparse.BooleanOptionalAction, default=None,
                        help="Include each available outcome in minibatches, with importance correction")
    parser.add_argument("--updates-per-round", type=int, default=None, help="PPO minibatch steps per model per completed wave; overrides saved setting")
    parser.add_argument("--parallel-games", type=int, default=None, help="Batched games per wave; overrides saved setting on resume")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--move-delay", type=float, default=None, help="Minimum wave/move interval; may override a resumed run")
    parser.add_argument("--max-plies", type=int, default=0, help="0: unlimited; cutoffs are discarded, not draws")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--resume", action="store_true", help="Restore config, models, Adam buffers and RNG from run-dir")
    args = parser.parse_args()
    if args.games < 0:
        parser.error("--games must be nonnegative")
    config = TrainingConfig(batch_size=8 if args.batch_size is None else args.batch_size, learning_rate=args.learning_rate,
                            move_delay=0.1 if args.move_delay is None else args.move_delay, max_plies=args.max_plies, seed=args.seed)
    model_config = ModelConfig(width=args.width, heads=args.heads, layers=args.layers, policy_width=args.policy_width)
    live = LiveState()
    server = make_server(live, args.port)

    def work():
        try:
            trainer = Trainer(live, args.run_dir, config, model_config, args.resume)
            if args.parallel_games is not None:
                trainer.config = replace(trainer.config, parallel_games=args.parallel_games)
            if args.move_delay is not None:
                trainer.config = replace(trainer.config, move_delay=args.move_delay)
            if args.updates_per_round is not None:
                trainer.config = replace(trainer.config, updates_per_round=args.updates_per_round)
            if args.batch_size is not None:
                trainer.config = replace(trainer.config, batch_size=args.batch_size)
            if args.outcome_stratified is not None:
                trainer.config = replace(trainer.config, outcome_stratified=args.outcome_stratified)
            live.publish(training_config=asdict(trainer.config))
            trainer.run(args.games)
        except Exception as error:
            traceback.print_exc()
            live.publish(phase="error", message=f"{type(error).__name__}: {error}")

    # Keep Metal runtime and its resource lifetime on the main thread.
    server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    server_thread.start()
    print(f"Watch: http://127.0.0.1:{server.server_port}  |  run: {args.run_dir.resolve()}", flush=True)
    try:
        work()
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("Stopping after the current computation and saving…", flush=True)
    finally:
        live.stop.set()
        server.shutdown()
        server.server_close()
        server_thread.join()


if __name__ == "__main__":
    main()
