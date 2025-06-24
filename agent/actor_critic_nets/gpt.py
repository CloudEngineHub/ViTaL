from typing import Dict
import flax.linen as nn
import jax
import jax.numpy as jnp

from agent.networks.gpt import GPT, GPTConfig
from agent.networks.mlp import MLP
from agent.policy_heads.deterministic_head import DeterministicHead
from utils import ActionType


class GPTActor(nn.Module):
    action_dim: int
    hidden_dim: int
    repr_dim: int
    action_type: ActionType

    def setup(self):
        self.action_token = self.param(
            "action_token", nn.initializers.normal(), (1, 1, self.repr_dim)
        )
        self.policy = GPT(
            GPTConfig(
                block_size=65,
                input_dim=self.repr_dim,
                output_dim=self.hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=self.hidden_dim,
                dropout=0.1,
            )
        )
        # LATERTODO: Currently hard-coded
        if self.action_type == ActionType.CONTINUOUS:
            self.action_head = DeterministicHead(
                hidden_dim=self.hidden_dim,
                output_dim=self.action_dim,
                num_layers=2,
                action_squash=True,
            )
        else:
            raise NotImplementedError

    def __call__(self, x: Dict[str, jax.Array], std: float, training: bool = False):
        x = jnp.stack(list(x.values()), axis=-2)
        B, T, C = x.shape
        action_token = jnp.tile(self.action_token, (B, 1, 1))
        x = jnp.concatenate([x, action_token], axis=1)

        x = self.policy(x, training=training)
        x = self.action_head(x[:, -1], std)
        return x


class GPTActorOffset(GPTActor):
    def setup(self):
        super().setup()
        self.base_act_projection = nn.Dense(self.repr_dim)

    def __call__(
        self,
        x: Dict[str, jax.Array],
        action_base: jax.Array,
        std: float,
        training: bool = False,
    ):
        obs = dict(base_act=self.base_act_projection(action_base))
        obs.update(x)
        return super().__call__(obs, std, training=training)


class GPTCritic(nn.Module):
    action_dim: int
    hidden_dim: int
    repr_dim: int
    num_obs_keys: int
    action_type: ActionType
    dropout_rate: float = 0.1

    def setup(self):
        if self.action_type != ActionType.CONTINUOUS:
            raise NotImplementedError("Only continuous actions are supported.")
        self.critic_token = self.param(
            "critic_token", nn.initializers.normal(), (1, 1, self.repr_dim)
        )
        self.Q1 = GPT(
            GPTConfig(
                block_size=65,
                input_dim=self.hidden_dim,
                output_dim=1,
                n_layer=8,
                n_head=4,
                n_embd=self.hidden_dim,
                dropout=0.1,
            )
        )
        self.Q2 = GPT(
            GPTConfig(
                block_size=65,
                input_dim=self.hidden_dim,
                output_dim=1,
                n_layer=8,
                n_head=4,
                n_embd=self.hidden_dim,
                dropout=0.1,
            )
        )

        self.base_act_projection = nn.Dense(self.repr_dim)
        self.offset_act_projection = nn.Dense(self.repr_dim)

        # self.Q1_value_project = MLP(hidden_dims=[self.hidden_dim] * 2, output_dim=1)
        # self.Q2_value_project = MLP(hidden_dims=[self.hidden_dim] * 2, output_dim=1)

        self.Q1_value_project = nn.Dense(1)
        self.Q2_value_project = nn.Dense(1)

        self.layer_norm_q1 = nn.LayerNorm()
        self.dropout_q1 = nn.Dropout(rate=self.dropout_rate)
        self.layer_norm_q2 = nn.LayerNorm()
        self.dropout_q2 = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, obs, action_base, offset_action, training: bool = False):
        B, T, C = x.shape
        critic_token = jnp.tile(self.critic_token, (B, 1, 1))
        base_act = self.base_act_projection(action_base)
        offset_act = self.offset_act_projection(offset_action)

        x = jnp.stack(list(obs.values()) + [base_act, offset_act], axis=-2)
        x = jnp.concatenate([x, critic_token], axis=1)

        q1_feat = self.Q1(x, training=training)[:, -1]
        q2_feat = self.Q2(x, training=training)[:, -1]

        q1_feat_norm = self.layer_norm_q1(q1_feat)
        q1_feat_drop = self.dropout_q1(q1_feat_norm, deterministic=not training)
        q2_feat_norm = self.layer_norm_q2(q2_feat)
        q2_feat_drop = self.dropout_q2(q2_feat_norm, deterministic=not training)

        q1 = self.Q1_value_project(q1_feat_drop)
        q2 = self.Q2_value_project(q2_feat_drop)
        return q1, q2
