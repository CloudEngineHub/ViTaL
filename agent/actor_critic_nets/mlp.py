from typing import Dict, Optional, Union
from jax import numpy as jnp
from flax import linen as nn
import jax
import tensorflow_probability.substrates.jax as tfp

from agent.policy_heads.deterministic_head import DeterministicHead

tfd = tfp.distributions

from agent.networks.mlp import MLP, EnsembleMLP
import jax.numpy as jnp
import distrax

from utils import ActionType


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(
        scale=scale, mode="fan_avg", distribution="uniform"
    )


class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(self, loc, scale_diag):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        super().__init__(distribution=distribution, bijector=distrax.Tanh())


class MLPActor(nn.Module):
    action_dim: int
    hidden_dim: int
    repr_dim: int
    action_type: ActionType
    std_parameterization: str = "exp"
    std_min: float = 1e-5
    std_max: float = 10.0
    tanh_squash_distribution: bool = True
    fixed_std: Optional[jnp.ndarray] = None
    init_final: Optional[float] = None

    def setup(self):
        self.policy = MLP(hidden_dims=[self.hidden_dim] * 2, output_dim=self.repr_dim)

        if self.init_final is not None:
            kernel_init = default_init(self.init_final)
        else:
            kernel_init = default_init()

        self.mean_layer = nn.Dense(self.action_dim, kernel_init=kernel_init)

        if self.fixed_std is None and self.std_parameterization not in [
            "fixed",
            "uniform",
        ]:
            self.std_layer = nn.Dense(self.action_dim, kernel_init=kernel_init)
        elif self.std_parameterization == "uniform":
            self.log_stds = self.param(
                "log_stds", nn.initializers.zeros, (self.action_dim,)
            )

    def __call__(
        self,
        x: Dict[str, jnp.ndarray],
        temperature: float = 1.0,
        training: bool = False,
    ) -> distrax.Distribution:
        x = jnp.concatenate(list(x.values()), axis=-1)
        features = self.policy(x, training=training)
        means = self.mean_layer(features)

        if self.fixed_std is not None:
            stds = jnp.broadcast_to(self.fixed_std, means.shape)
        elif self.std_parameterization == "exp":
            log_stds = self.std_layer(features)
            stds = jnp.exp(log_stds)
        elif self.std_parameterization == "softplus":
            raw_stds = self.std_layer(features)
            stds = nn.softplus(raw_stds)
        elif self.std_parameterization == "uniform":
            stds = jnp.exp(self.log_stds)
            stds = jnp.broadcast_to(stds, means.shape)
        else:
            raise ValueError(
                f"Invalid std_parameterization: {self.std_parameterization}"
            )

        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution and self.action_type == ActionType.CONTINUOUS:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution


class MLPActorOffsetSAC(nn.Module):
    action_dim: int
    hidden_dim: int
    repr_dim: int
    action_type: int
    num_obs_keys: int
    std_parameterization: str = "exp"
    std_min: float = 1e-5
    std_max: float = 1.0
    tanh_squash_distribution: bool = True
    init_final: float = 0.01
    dropout_rate: Optional[float] = None
    use_layer_norm: bool = False

    def setup(self):
        self.net = MLP(
            hidden_dims=[self.hidden_dim] * 3,
            output_dim=self.repr_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
        )

        self.mean_layer = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.uniform(scale=self.init_final),
        )
        self.log_std_layer = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.uniform(scale=self.init_final),
        )

    def __call__(
        self,
        obs: Dict[str, jax.Array],
        action_base: jax.Array,
        temperature: float = 1.0,
        training: bool = False,
    ) -> distrax.Distribution:
        x = []
        for key, arr in obs.items():
            arr_flat = jnp.reshape(arr, (arr.shape[0], -1))
            x.append(arr_flat)
        x.append(action_base)
        x = jnp.concatenate(x, axis=-1)

        h = self.net(x, training=training)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)

        if self.std_parameterization == "exp":
            std = jnp.exp(log_std)
        elif self.std_parameterization == "softplus":
            std = jax.nn.softplus(log_std)
        elif self.std_parameterization == "uniform":
            std = jnp.ones_like(mean) * self.std_min
        else:
            raise ValueError(
                f"Unknown std parameterization: {self.std_parameterization}"
            )

        std = jnp.clip(std, self.std_min, self.std_max)
        std = std * jnp.sqrt(temperature)

        if self.tanh_squash_distribution:
            # Tanh-squashed Gaussian for bounded action spaces
            base_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
            dist = distrax.Transformed(
                distribution=base_dist,
                bijector=distrax.Block(distrax.Tanh(), ndims=1),
            )
        else:
            # Standard Gaussian
            dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

        return dist


class MLPActorOffsetDDPG(nn.Module):
    action_dim: int
    hidden_dim: int
    repr_dim: int
    action_type: int
    num_obs_keys: int
    init_final: float = 0.01
    dropout_rate: Optional[float] = None
    use_layer_norm: bool = False

    def setup(self):
        self.net = MLP(
            hidden_dims=[self.hidden_dim] * 3,
            output_dim=self.repr_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
        )

        self.action_head = DeterministicHead(
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
            num_layers=2,
            action_squash=True,
        )

    def __call__(
        self,
        obs: Dict[str, jax.Array],
        action_base: jax.Array,
        noise_scale: float = 1.0,
        training: bool = False,
    ) -> jnp.ndarray:
        x = []
        for key, arr in obs.items():
            arr_flat = jnp.reshape(arr, (arr.shape[0], -1))
            x.append(arr_flat)
        x.append(action_base)
        x = jnp.concatenate(x, axis=-1)

        h = self.net(x, training=training)
        return self.action_head(h, noise_scale)


class MLPCriticSAC(nn.Module):
    action_dim: int
    hidden_dim: int
    repr_dim: int
    num_obs_keys: int
    action_type: ActionType
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    ensemble_size: int = 10

    def setup(self):
        if self.action_type != ActionType.CONTINUOUS:
            raise NotImplementedError("Only continuous actions supported")

        self.Q_ensemble = EnsembleMLP(
            ensemble_size=self.ensemble_size,
            hidden_dims=[self.hidden_dim] * 2,
            output_dim=1,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )

        self.base_act_projection = nn.Dense(self.hidden_dim * self.num_obs_keys)
        self.offset_act_projection = nn.Dense(self.hidden_dim * self.num_obs_keys)

    def __call__(
        self,
        obs: Dict[str, jnp.ndarray],
        action_base: jnp.ndarray,
        offset_action: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        base_act_features = self.base_act_projection(action_base)
        offset_act_features = self.offset_act_projection(offset_action)

        obs_features = list(obs.values())
        x = jnp.concatenate(
            obs_features + [base_act_features, offset_act_features], axis=-1
        )

        q_values = self.Q_ensemble(x, training=training)
        q_values = jnp.squeeze(q_values, axis=-1)
        return q_values


class MLPCriticDDPG(nn.Module):
    action_dim: int
    hidden_dim: int
    repr_dim: int
    num_obs_keys: int
    action_type: ActionType
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    def setup(self):
        if self.action_type != ActionType.CONTINUOUS:
            raise NotImplementedError("Only continuous actions supported")

        # DDPG uses exactly 2 Q-networks
        self.Q_ensemble = EnsembleMLP(
            ensemble_size=2,
            hidden_dims=[self.hidden_dim] * 2,
            output_dim=1,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )

        self.base_act_projection = nn.Dense(self.hidden_dim * self.num_obs_keys)
        self.offset_act_projection = nn.Dense(self.hidden_dim * self.num_obs_keys)

    def __call__(
        self,
        obs: Dict[str, jnp.ndarray],
        action_base: jnp.ndarray,
        offset_action: jnp.ndarray,
        training: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        base_act_features = self.base_act_projection(action_base)
        offset_act_features = self.offset_act_projection(offset_action)

        obs_features = list(obs.values())
        x = jnp.concatenate(
            obs_features + [base_act_features, offset_act_features], axis=-1
        )

        q_values = self.Q_ensemble(x, training=training)
        q_values = jnp.squeeze(q_values, axis=-1)

        # Return exactly two Q-values for DDPG
        return q_values[0], q_values[1]
