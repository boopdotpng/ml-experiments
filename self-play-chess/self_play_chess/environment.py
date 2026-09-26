"""Exact move legality and game history are owned by python-chess."""
import chess
import numpy as np

PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
CLAIM_DRAW = 5 * 64 * 64
ACTION_COUNT = CLAIM_DRAW + 1


def move_to_action(move: chess.Move) -> int:
    if not move or move.drop is not None:
        raise ValueError("Null moves and drops are not chess actions")
    return PROMOTIONS.index(move.promotion) * 4096 + move.from_square * 64 + move.to_square


def action_to_move(action: int) -> chess.Move:
    if not 0 <= action < CLAIM_DRAW:
        raise ValueError("Not a board-move action")
    promotion, squares = divmod(int(action), 4096)
    source, destination = divmod(squares, 64)
    return chess.Move(source, destination, promotion=PROMOTIONS[promotion])


class ChessEnv:
    def __init__(self, board: chess.Board | None = None):
        self.board = board.copy(stack=True) if board is not None else chess.Board()
        if self.board.chess960 or self.board.uci_variant != "chess" or not self.board.is_valid():
            raise ValueError("Only valid standard chess positions are supported")
        self.draw_claimed = False

    def outcome(self) -> chess.Outcome | None:
        if self.draw_claimed:
            return chess.Outcome(chess.Termination.THREEFOLD_REPETITION
                if self.board.can_claim_threefold_repetition() else chess.Termination.FIFTY_MOVES, None)
        return self.board.outcome(claim_draw=False)

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        if self.outcome() is None:
            for move in self.board.legal_moves:
                mask[move_to_action(move)] = True
            mask[CLAIM_DRAW] = self.board.can_claim_draw()
        return mask

    def step(self, action: int) -> chess.Outcome | None:
        if not isinstance(action, (int, np.integer)) or not 0 <= action < ACTION_COUNT:
            raise ValueError("Invalid action ID")
        if not self.legal_mask()[action]:
            raise ValueError("Action is illegal or the game has ended")
        if action == CLAIM_DRAW:
            self.draw_claimed = True
        else:
            self.board.push(action_to_move(action))
        return self.outcome()

    def terminal_value(self, color: chess.Color) -> float:
        outcome = self.outcome()
        if outcome is None:
            raise ValueError("Game has not ended")
        return 0.0 if outcome.winner is None else (1.0 if outcome.winner == color else -1.0)
