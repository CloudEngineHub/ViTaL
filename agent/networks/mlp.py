from typing import Callable, List, Optional, Sequence
import flax.linen as nn
from jax import numpy as jnp
import jax


class MLP(nn.Module):
    hidden_dims: List[int]
    output_dim: int
    dropout_rate: Optional[float] = None
    use_layer_norm: bool = False
    activation_layer: Callable[..., nn.Module] = nn.silu
    kernel_init: Callable = nn.initializers.orthogonal
    bias_init: Callable = nn.initializers.zeros_init

    def setup(self):
        self.layers = [
            nn.Dense(
                hidden_dim,
                kernel_init=self.kernel_init(),
                bias_init=self.bias_init(),
            )
            for hidden_dim in self.hidden_dims
        ]

        if self.dropout_rate:
            self.dropouts = [
                nn.Dropout(rate=self.dropout_rate) for _ in self.hidden_dims
            ]
        else:
            self.dropouts = [None] * len(self.hidden_dims)

        if self.use_layer_norm:
            self.norms = [nn.LayerNorm() for _ in self.hidden_dims]
        else:
            self.norms = [None] * len(self.hidden_dims)

        self.output_layer = nn.Dense(
            self.output_dim,
            kernel_init=self.kernel_init(),
            bias_init=self.bias_init(),
        )

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        for dense, dropout, norm in zip(self.layers, self.dropouts, self.norms):
            x = dense(x)
            if dropout is not None:
                x = dropout(x, deterministic=not training)
            if norm is not None:
                x = norm(x)
            x = self.activation_layer(x)
        return self.output_layer(x)


class EnsembleMLP(nn.Module):
    ensemble_size: int
    hidden_dims: Sequence[int]
    output_dim: int
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        ensemble_outputs = []

        for ensemble_idx in range(self.ensemble_size):
            h = x
            for i, hidden_dim in enumerate(self.hidden_dims):
                h = nn.Dense(hidden_dim, name=f"dense_{i}_ensemble_{ensemble_idx}")(h)

                if i < len(self.hidden_dims) - 1 or self.activate_final:
                    h = nn.relu(h)

                    if self.use_layer_norm:
                        h = nn.LayerNorm(name=f"ln_{i}_ensemble_{ensemble_idx}")(h)

                    if self.dropout_rate is not None and training:
                        h = nn.Dropout(
                            rate=self.dropout_rate,
                            name=f"dropout_{i}_ensemble_{ensemble_idx}",
                        )(h, deterministic=not training)

            h = nn.Dense(self.output_dim, name=f"output_ensemble_{ensemble_idx}")(h)

            ensemble_outputs.append(h)

        # Stack outputs: (ensemble_size, batch_size, output_dim)
        return jnp.stack(ensemble_outputs, axis=0)
