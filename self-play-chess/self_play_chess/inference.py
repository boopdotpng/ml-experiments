"""Fixed-shape TinyJit inference; recreate after optimizer updates."""
from tinygrad import TinyJit
from .encoding import Observation


class Inference:
    def __init__(self, model, jit=True):
        def forward(pieces, turn, en_passant, state, legal):
            logits, values = model(Observation(pieces, turn, en_passant, state, legal))
            # Caller supplies only nonterminal boards; environment validates actions too.
            return legal.where(logits, -1e9).softmax(axis=-1).realize(), values.softmax(axis=-1).realize()
        self.forward = TinyJit(forward) if jit else forward

    def __call__(self, observation):
        args = [getattr(observation, name).contiguous().realize() for name in Observation.__dataclass_fields__]
        policy, value = self.forward(*args)
        return policy.numpy().copy(), value.numpy().copy()
