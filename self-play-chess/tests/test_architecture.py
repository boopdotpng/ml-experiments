import unittest

import chess
import numpy as np
from tinygrad import Tensor, nn, Context
from tinygrad.nn.state import get_parameters

from self_play_chess.encoding import encode
from self_play_chess.environment import ACTION_COUNT, CLAIM_DRAW, ChessEnv, action_to_move, move_to_action
from self_play_chess.model import ChessTransformer, ModelConfig, legal_policy


class RulesTests(unittest.TestCase):
    def test_initial_encoding_and_illegal_action(self):
        env = ChessEnv()
        observation = encode([env])
        np.testing.assert_array_equal(observation.state.numpy()[0, :4], [1, 1, 1, 1])
        self.assertEqual(observation.pieces.numpy()[0, chess.A1], chess.ROOK)
        self.assertEqual(observation.pieces.numpy()[0, chess.E8], chess.KING + 6)
        self.assertEqual(int(observation.legal.numpy().sum()), 20)
        before = env.board.fen()
        with self.assertRaises(ValueError):
            env.step(move_to_action(chess.Move.from_uci("e2e5")))
        self.assertEqual(env.board.fen(), before)

    def test_special_moves(self):
        cases = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", chess.F1, chess.ROOK),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", chess.D8, chess.ROOK),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", chess.D6, chess.PAWN),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", chess.A8, chess.KNIGHT),
        ]
        for fen, uci, square, piece in cases:
            with self.subTest(uci=uci):
                env = ChessEnv(chess.Board(fen))
                move = chess.Move.from_uci(uci)
                action = move_to_action(move)
                self.assertEqual(action_to_move(action), move)
                env.step(action)
                self.assertEqual(env.board.piece_type_at(square), piece)
                if uci == "e5d6":
                    self.assertIsNone(env.board.piece_at(chess.D5))
        env = ChessEnv(chess.Board(cases[-1][0]))
        promotions = [m for m in env.board.legal_moves if m.promotion]
        self.assertEqual(len({move_to_action(m) for m in promotions}), 4)

    def test_repetition_claim_and_history_copy(self):
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            board.push_uci(uci)
        env = ChessEnv(board)
        self.assertTrue(env.legal_mask()[CLAIM_DRAW])
        self.assertAlmostEqual(float(encode([env]).state.numpy()[0, 5]), 3 / 5)
        self.assertIsNone(env.outcome())  # claimable is not automatic
        env.step(CLAIM_DRAW)
        self.assertEqual(env.terminal_value(chess.WHITE), 0)
        self.assertFalse(env.legal_mask().any())
        self.assertEqual(len(board.move_stack), 8)

    def test_terminal_and_fifty_move_claim(self):
        env = ChessEnv()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            env.step(move_to_action(chess.Move.from_uci(uci)))
        self.assertEqual(env.terminal_value(chess.WHITE), -1)
        self.assertEqual(env.terminal_value(chess.BLACK), 1)
        self.assertFalse(env.legal_mask().any())
        with self.assertRaises(ValueError):
            legal_policy(Tensor.zeros(1, ACTION_COUNT), encode([env]).legal)
        env = ChessEnv(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 51"))
        self.assertTrue(env.legal_mask()[CLAIM_DRAW])
        env.step(CLAIM_DRAW)
        self.assertEqual(env.outcome().termination, chess.Termination.FIFTY_MOVES)


class ModelTests(unittest.TestCase):
    def test_batched_forward_mask_and_optimizer_step(self):
        Tensor.manual_seed(7)
        envs = [ChessEnv(), ChessEnv(chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"))]
        observation = encode(envs)
        model = ChessTransformer(ModelConfig(width=32, heads=4, layers=1, policy_width=8))
        parameters = get_parameters(model)
        optimizer = nn.optim.Adam(parameters, lr=1e-3)
        before = model.piece.weight.numpy().copy()
        with Context(TRAINING=1):
            logits, values = model(observation)
            self.assertEqual(logits.shape, (2, ACTION_COUNT))
            self.assertEqual(values.shape, (2, 3))
            policy = legal_policy(logits, observation.legal)
            probabilities = policy.numpy()
            mask = observation.legal.numpy()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            self.assertTrue((probabilities[~mask] == 0).all())
            self.assertTrue(np.isfinite(values.numpy()).all())
            # Arbitrary labels ONLY check differentiability; these are not training data.
            targets = Tensor(np.array([[np.flatnonzero(row)[0]] for row in mask], dtype=np.int32))
            masked = observation.legal.where(logits, -1e9)
            loss = -masked.log_softmax(axis=-1).gather(1, targets).mean()
            loss = loss + values.sparse_categorical_crossentropy(Tensor([2, 1]))
            optimizer.zero_grad()
            loss.backward()
            for parameter in parameters:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(np.isfinite(parameter.grad.numpy()).all())
            optimizer.step()
        self.assertFalse(np.array_equal(before, model.piece.weight.numpy()))


if __name__ == "__main__":
    unittest.main()
