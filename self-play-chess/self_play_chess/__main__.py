"""Short untrained rollout to exercise the architecture, not a training run."""
import argparse

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters

from .encoding import encode
from .environment import CLAIM_DRAW, ChessEnv, action_to_move
from .model import ChessTransformer, ModelConfig, legal_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.plies < 0:
        parser.error("--plies must be nonnegative")
    Tensor.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ChessTransformer(ModelConfig())
    env = ChessEnv()
    print(f"Randomly initialized model: {sum(p.numel() for p in get_parameters(model)):,} parameters")
    for ply in range(args.plies):
        if env.outcome() is not None:
            break
        observation = encode([env])
        logits, _ = model(observation)
        probabilities = legal_policy(logits, observation.legal).numpy()[0].astype(np.float64)
        probabilities /= probabilities.sum()
        action = int(rng.choice(len(probabilities), p=probabilities))
        print(f"{ply + 1}: {'claim draw' if action == CLAIM_DRAW else action_to_move(action).uci()}")
        env.step(action)
    print(env.board)
    print(f"Result: {env.outcome().result() if env.outcome() else 'unfinished (demo ply limit)'}")


if __name__ == "__main__":
    main()
