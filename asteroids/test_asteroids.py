import numpy as np
import torch

from asteroids import Agent, compute_gae, episode_stats, signed_sqrt


def test_agent_shapes_and_gradients():
    agent = Agent()
    obs = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    action, logprob, entropy, value = agent.get_action_and_value(obs)
    assert action.shape == logprob.shape == entropy.shape == value.shape == (3,)
    (-logprob.mean() + value.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in agent.parameters())


def test_gae_stops_at_terminal():
    rewards = torch.tensor([[1.0], [2.0], [100.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    advantages, returns = compute_gae(rewards, values, dones, torch.tensor([0.0]), 1.0, 1.0)
    assert torch.equal(advantages, torch.tensor([[3.0], [2.0], [100.0]]))
    assert torch.equal(returns, advantages)


def test_reward_transform():
    transformed = signed_sqrt(np.array([-100.0, 0.0, 25.0]))
    np.testing.assert_array_equal(transformed, [-10.0, 0.0, 5.0])


def test_vector_episode_info():
    infos = {
        "final_info": {
            "episode": {"r": np.array([10.0, 20.0]), "l": np.array([3, 4])},
            "_episode": np.array([True, False]),
        }
    }
    assert episode_stats(infos) == [(10.0, 3)]
