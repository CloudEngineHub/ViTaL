import jax
import jax.numpy as jnp
import flax.linen as nn

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class DeterministicHead(nn.Module):
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    action_squash: bool = True

    def setup(self):
        layer_sizes = [self.hidden_dim] * self.num_layers + [self.output_dim]
        self.layers = [nn.Dense(size) for size in layer_sizes]

    def __call__(self, x: jax.Array, std: float = 0.1) -> tfd.Distribution:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.layers[-1](x)
        if self.action_squash:
            x = nn.tanh(x)
        x = tfd.TruncatedNormal(loc=x, scale=std * jnp.ones_like(x), low=-1.0, high=1.0)
        return x
