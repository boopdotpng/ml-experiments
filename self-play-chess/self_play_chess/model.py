"""Bidirectional board transformer; no pretrained weights or strategic features."""
from dataclasses import dataclass
import math

from tinygrad import Tensor, nn

from .encoding import Observation, STATE_FEATURES
from .environment import CLAIM_DRAW


@dataclass(frozen=True)
class ModelConfig:
    width: int = 128
    heads: int = 4
    layers: int = 4
    ff_multiplier: int = 4
    policy_width: int = 32

    def __post_init__(self):
        if min(self.width, self.heads, self.layers, self.ff_multiplier, self.policy_width) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class Block:
    def __init__(self, config: ModelConfig):
        d = config.width
        self.heads = config.heads
        self.attention_norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attention_out = nn.Linear(d, d, bias=False)
        self.ff_norm = nn.LayerNorm(d)
        self.ff_in = nn.Linear(d, config.ff_multiplier * d)
        self.ff_out = nn.Linear(config.ff_multiplier * d, d)

    def __call__(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(self.attention_norm(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        attention = q.scaled_dot_product_attention(k, v, is_causal=False)
        x = x + self.attention_out(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.ff_out(self.ff_in(self.ff_norm(x)).gelu())


class ChessTransformer:
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        d = config.width
        self.piece = nn.Embedding(13, d)
        self.rank = nn.Embedding(8, d)
        self.file = nn.Embedding(8, d)
        self.turn = nn.Embedding(2, d)
        self.en_passant = nn.Embedding(65, d)
        self.state = nn.Linear(STATE_FEATURES, d)
        self.blocks = [Block(config) for _ in range(config.layers)]
        self.norm = nn.LayerNorm(d)
        # Separate source/destination vectors for each promotion category.
        self.source = nn.Linear(d, 5 * config.policy_width)
        self.destination = nn.Linear(d, 5 * config.policy_width)
        self.claim = nn.Linear(d, 1)
        self.value = nn.Linear(d, 3)  # loss, draw, win, from side-to-move perspective

    def __call__(self, observation: Observation) -> tuple[Tensor, Tensor]:
        batch = observation.pieces.shape[0]
        squares = Tensor.arange(64).to(observation.pieces.device)
        board = self.piece(observation.pieces) + self.rank(squares // 8) + self.file(squares % 8)
        state = self.state(observation.state) + self.turn(observation.turn) + self.en_passant(observation.en_passant)
        x = state.unsqueeze(1).cat(board, dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        board, state = x[:, 1:], x[:, 0]
        shape = (batch, 64, 5, self.config.policy_width)
        source = self.source(board).reshape(shape).permute(0, 2, 1, 3)
        destination = self.destination(board).reshape(shape).permute(0, 2, 3, 1)
        moves = (source @ destination / math.sqrt(self.config.policy_width)).reshape(batch, CLAIM_DRAW)
        return moves.cat(self.claim(state), dim=1), self.value(state)


def legal_policy(logits: Tensor, legal: Tensor) -> Tensor:
    """Only call on nonterminal rows: an empty legal set has no policy."""
    if not legal.any(axis=1).all().item():
        raise ValueError("Cannot choose an action for a terminal position")
    return legal.where(logits, float("-inf")).softmax(axis=-1)
