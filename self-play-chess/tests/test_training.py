import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import chess
import numpy as np
from tinygrad import Tensor, Context
from tinygrad.nn.state import get_state_dict

from self_play_chess.encoding import Observation, encode
from self_play_chess.environment import ChessEnv, move_to_action
from self_play_chess.model import ModelConfig
from self_play_chess.train import make_server
from self_play_chess.training import LiveState, Trainer, TrainingConfig, actor_critic_loss, outcome_target, sample_training_indices
from self_play_chess.inference import Inference
from self_play_chess.model import ChessTransformer


class TrainingTests(unittest.TestCase):
    def test_stratified_sampling_preserves_outcome_distribution(self):
        classes = np.array([0]*2 + [1]*94 + [2]*4)
        indices, weights = sample_training_indices(classes, 32, 4, np.random.default_rng(7), True)
        for start in range(0, 128, 32):
            chosen, w = classes[indices[start:start+32]], weights[start:start+32]
            self.assertEqual(set(chosen), {0, 1, 2})
            np.testing.assert_allclose([w[chosen == label].sum()/32 for label in range(3)],
                                       [.02, .94, .04], atol=1e-6)
        indices, weights = sample_training_indices([1]*5, 8, 2, np.random.default_rng(7), True)
        self.assertTrue(np.all(indices < 5))
        np.testing.assert_array_equal(weights, np.ones(16))

    def test_weighted_loss_matches_uniform_objective(self):
        # Class-constant rows make stratification's correction exact, including gradients.
        with Context(TRAINING=1):
            def compute(labels, weights):
                logits = Tensor([[.3, -.2]]).is_param_(True)
                values = Tensor([[.1, .2, -.1]]).is_param_(True)
                n = len(labels)
                result = actor_critic_loss(logits.expand(n, 2), values.expand(n, 3),
                    Tensor.ones(n, 2).bool(), Tensor([0]*n), Tensor([float(x-1) for x in labels]),
                    Tensor(labels), sample_weights=Tensor(weights))
                result[0].backward()
                return np.array([x.item() for x in result]), logits.grad.numpy(), values.grad.numpy()
            full = compute([0]+[1]*8+[2], [1.]*10)
            balanced = compute([0, 1, 2], [.3, 2.4, .3])
            for expected, actual in zip(full, balanced):
                np.testing.assert_allclose(expected, actual, atol=1e-6)

    def test_ppo_clips_improvements_outside_ratio_range(self):
        with Context(TRAINING=1):
            logits = Tensor([[np.log(9.), 0.], [-np.log(9.), 0.], [0., 0.], [0., 0.]]).is_param_(True)
            advantages = Tensor([1., -1., 1., -1.])
            loss, *_ = actor_critic_loss(logits, Tensor.zeros(4, 3), Tensor.ones(4, 2).bool(),
                Tensor([0, 0, 0, 0]), advantages, Tensor([2, 0, 2, 0]), 0, 0,
                old_log_probs=Tensor([np.log(0.5)]*4), advantages=advantages)
            loss.backward()
            gradients = logits.grad.numpy()
            np.testing.assert_allclose(gradients[:2], 0, atol=1e-6)
            self.assertLess(gradients[2, 0], 0)
            self.assertGreater(gradients[3, 0], 0)

    def test_jit_matches_eager_with_changing_inputs(self):
        model = ChessTransformer(ModelConfig(width=32, layers=1, policy_width=8))
        eager, jit = Inference(model, jit=False), Inference(model)
        for uci in [None, "e2e4", "d2d4", "g1f3"]:
            envs = [ChessEnv() for _ in range(4)]
            if uci:
                for env in envs:
                    env.board.push_uci(uci)
            expected, _ = eager(encode(envs))
            actual, _ = jit(encode(envs))
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
            self.assertTrue((actual[~encode(envs).legal.numpy()] == 0).all())

    def test_parallel_cutoffs_and_legality(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(LiveState(), Path(directory), TrainingConfig(parallel_games=4, max_plies=4, move_delay=0))
            self.assertTrue(trainer.play_parallel(4))
            self.assertEqual(trainer.games, 4)
            self.assertEqual(trainer.stats["truncated"], 4)
            self.assertEqual(trainer.updates, 0)
            self.assertTrue(all(row["plies"] == 4 for row in trainer.live.snapshot()["history"]))

    def test_outcome_perspectives_and_policy_gradient_direction(self):
        outcome = chess.Outcome(chess.Termination.CHECKMATE, chess.WHITE)
        self.assertEqual(outcome_target(outcome, chess.WHITE), (1, 2))
        self.assertEqual(outcome_target(outcome, chess.BLACK), (-1, 0))
        self.assertEqual(outcome_target(chess.Outcome(chess.Termination.STALEMATE, None), chess.WHITE), (0, 1))
        with Context(TRAINING=1):
            logits = Tensor([[0., 0.], [0., 0.]]).is_param_(True)
            loss, *_ = actor_critic_loss(logits, Tensor.zeros(2, 3), Tensor([[True, True], [True, True]]),
                                         Tensor([0, 0]), Tensor([1., -1.]), Tensor([2, 0]), 0, 0)
            loss.backward()
            gradient = logits.grad.numpy()
            self.assertLess(gradient[0, 0], 0)  # descent increases winning action logit
            self.assertGreater(gradient[1, 0], 0)  # decreases losing action logit

    def test_two_learners_update_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            # Exercise the exact default Mac launch configuration, not only a toy shape.
            trainer = Trainer(LiveState(), Path(directory), TrainingConfig(updates_per_round=3, outcome_stratified=True))
            env, trajectories = ChessEnv(), {"A": [], "B": []}
            before = {name: model.piece.weight.numpy().copy() for name, model in trainer.models.items()}
            self.assertFalse(np.array_equal(before["A"], before["B"]))
            # A short legal terminal trajectory tests training plumbing, not pretraining.
            for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
                observation = encode([env])
                action = move_to_action(chess.Move.from_uci(uci))
                name = "A" if env.board.turn else "B"
                arrays = {field: getattr(observation, field).numpy().copy() for field in Observation.__dataclass_fields__}
                trajectories[name].append((arrays, action, env.board.turn))
                env.step(action)
            for name in ("A", "B"):
                metrics = trainer.update(name, trajectories[name], env.outcome())
                self.assertEqual(metrics["samples"], 24)
                self.assertEqual(metrics["optimizer_steps"], 3)
                self.assertTrue(np.isfinite(list(metrics.values())).all())
                self.assertFalse(np.array_equal(before[name], trainer.models[name].piece.weight.numpy()))
            trainer.games = 3
            trainer.save()
            restored = Trainer(LiveState(), Path(directory), resume=True)
            self.assertEqual(restored.games, 3)
            self.assertEqual(restored.updates, 6)
            self.assertEqual(restored.training_samples, 48)
            self.assertEqual(restored.model_config, trainer.model_config)
            original = get_state_dict(trainer.state_objects())
            loaded = get_state_dict(restored.state_objects())
            for key in original:
                np.testing.assert_array_equal(original[key].numpy(), loaded[key].numpy())
            self.assertEqual(trainer.rng.random(), restored.rng.random())
            # Loaded optimizer state must be usable for another update.
            restored.update("A", trajectories["A"], env.outcome())
            self.assertEqual(restored.updates, 9)
            self.assertEqual(restored.training_samples, 72)

    def test_cutoff_is_not_draw_or_training_target(self):
        with tempfile.TemporaryDirectory() as directory:
            live = LiveState()
            trainer = Trainer(live, Path(directory), TrainingConfig(max_plies=1, move_delay=0),
                              ModelConfig(width=16, heads=2, layers=1, policy_width=4))
            self.assertTrue(trainer.play_game())
            self.assertEqual(trainer.updates, 0)
            self.assertEqual(trainer.stats["draws"], 0)
            self.assertEqual(trainer.stats["truncated"], 1)
            self.assertEqual(live.snapshot()["history"][0]["result"], "*")

    def test_local_server_controls(self):
        live = LiveState()
        server = make_server(live, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(url) as response:
                self.assertIn(b"Learning across the board", response.read())
            with urlopen(url + "/api/state") as response:
                self.assertEqual(json.load(response)["phase"], "starting")
            with self.assertRaises(HTTPError):
                urlopen(Request(url + "/api/stop", method="POST"))
            for action, event in [("pause", live.pause), ("resume", live.pause), ("stop", live.stop)]:
                with urlopen(Request(url + "/api/" + action, method="POST",
                             headers={"Origin": url, "X-Chess-Control": "1"})) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(event.is_set(), action != "resume")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
