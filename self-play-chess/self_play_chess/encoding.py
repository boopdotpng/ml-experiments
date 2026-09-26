"""a1..h1, a2..h2, ... a8..h8. IDs: empty=0, white=1..6, black=7..12."""
from dataclasses import dataclass

import chess
import numpy as np
from tinygrad import Tensor

from .environment import ChessEnv

# WK, WQ, BK, BQ, halfmove / 150, current repetition / 5,
# threefold claim available, fifty-move claim available.
STATE_FEATURES = 8


@dataclass
class Observation:
    pieces: Tensor       # [B, 64], integer categories
    turn: Tensor         # [B], black=0, white=1
    en_passant: Tensor   # [B], square=0..63, none=64
    state: Tensor        # [B, 8], explicit numeric features
    legal: Tensor        # [B, ACTION_COUNT], bool; terminal rows are all false


def encode(envs: list[ChessEnv]) -> Observation:
    if not envs:
        raise ValueError("Cannot encode an empty batch")
    pieces, turns, eps, states, masks = [], [], [], [], []
    for env in envs:
        board = env.board
        row = np.zeros(64, dtype=np.int32)
        for square, piece in board.piece_map().items():
            row[square] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        repetitions = next(n for n in range(5, 0, -1) if board.is_repetition(n))
        pieces.append(row)
        turns.append(int(board.turn))
        eps.append(board.ep_square if board.ep_square is not None else 64)
        states.append([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
            min(board.halfmove_clock, 150) / 150,
            repetitions / 5,
            board.can_claim_threefold_repetition(),
            board.can_claim_fifty_moves(),
        ])
        masks.append(env.legal_mask())
    return Observation(Tensor(np.array(pieces, dtype=np.int32)),
        Tensor(np.array(turns, dtype=np.int32)), Tensor(np.array(eps, dtype=np.int32)),
        Tensor(np.array(states, dtype=np.float32)), Tensor(np.array(masks)))
