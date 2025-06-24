from typing import Tuple
import flax.linen as nn
from jax import numpy as jnp


class Encoder(nn.Module):
    repr_dim: int
    obs_shape: Tuple[int, int, int]

    @nn.compact
    def __call__(self, obs, *args, **kwargs):
        # obs = 2 * (obs - 0.5)  # / 255.0 - 0.5
        obs = obs - 0.5
        # TODO: Verify conv net init
        obs = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=0,
        )(obs)
        obs = nn.relu(obs)
        for _ in range(3):
            obs = nn.Conv(
                features=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=0,
            )(obs)
            obs = nn.relu(obs)
        obs = jnp.reshape(obs, (obs.shape[0], -1))
        obs = nn.Dense(self.repr_dim)(obs)
        obs = nn.LayerNorm()(obs)
        obs = nn.tanh(obs)
        return obs
