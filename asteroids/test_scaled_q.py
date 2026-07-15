import numpy as np
import torch

from scaled_q import (
    ScaledAgent,
    SequenceReplay,
    n_step_returns,
    quantile_huber,
    stacked_observations,
)


def test_compact_frames_reconstruct_stacks():
    frames = torch.arange(7).view(1, 7, 1, 1).expand(1, 7, 2, 2)
    obs = stacked_observations(frames)
    assert obs.shape == (1, 4, 4, 2, 2)
    assert torch.equal(obs[0, 0, :, 0, 0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(obs[0, 3, :, 0, 0], torch.tensor([3, 4, 5, 6]))


def test_n_step_returns_stop_at_done():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    dones = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    returns, bootstrap = n_step_returns(rewards, dones, torch.tensor([0, 2]), 2, 1.0)
    assert torch.equal(returns, torch.tensor([[3.0, 7.0]]))
    assert torch.equal(bootstrap, torch.tensor([[0.0, 1.0]]))


def test_scaled_agent_shapes_and_distributional_gradient():
    agent = ScaledAgent(width=1, hidden=64, cosines=8)
    obs = torch.randint(0, 256, (2, 5, 4, 84, 84), dtype=torch.uint8)
    features = agent.sequence_features(obs, burn_in=2)
    quantiles, taus = agent.score_head(features, 4)
    assert features.shape == (2, 3, 64)
    assert quantiles.shape == (2, 3, 4, 14)
    assert taus.shape == (2, 3, 4)
    prediction = quantiles[..., 0].reshape(-1, 4)
    loss = quantile_huber(prediction, torch.zeros_like(prediction), taus.reshape(-1, 4)).mean()
    loss.backward()
    assert agent.encoder[0].conv.weight.grad is not None


def test_sequence_replay_round_trip():
    replay = SequenceReplay(capacity=3, sequence_length=4)
    frames = np.zeros((8, 2, 84, 84), dtype=np.uint8)
    frames[:, 1] = 1
    values = np.arange(8).reshape(4, 2)
    replay.add_batch(
        frames=frames,
        actions=values,
        rewards=values,
        costs=np.zeros_like(values),
        dones=np.zeros_like(values, dtype=bool),
    )
    batch, indices = replay.sample(2, beta=0.4, device=torch.device("cpu"))
    assert replay.size == 2
    assert batch["frames"].shape == (2, 8, 84, 84)
    assert batch["actions"].shape == (2, 4)
    replay.update_priorities(indices, np.array([2.0, 3.0]))
