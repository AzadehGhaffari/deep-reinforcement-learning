from __future__ import annotations

import numpy as np

from rl.replay_buffer import ReplayBuffer


def test_replay_buffer_sampling_shapes() -> None:
    buffer = ReplayBuffer(capacity=10)
    for _ in range(6):
        state = np.zeros(8, dtype=np.float32)
        next_state = np.ones(8, dtype=np.float32)
        buffer.push(state, 1, 0.5, next_state, False)

    states, actions, rewards, next_states, dones = buffer.sample(4)
    assert states.shape == (4, 8)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 8)
    assert dones.shape == (4,)