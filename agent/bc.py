from functools import partial
from flax import struct
from typing import Callable, Dict, List, Tuple, Union
import dm_env
import dm_env.specs
import gym
import jax
from jax import numpy as jnp
import optax
from flax.traverse_util import path_aware_map

import numpy as np

from agent.actor_critic_nets.mlp import MLPActor
from agent.actor_critic_nets.gpt import GPTActor
from agent.agent import Agent
from flax.training.train_state import TrainState

from agent.networks.encoder import Encoder
from agent.networks.mlp import MLP
from agent.networks.resnet_encoder import ResNet18Encoder
from utils import ActionType, ActorCriticType, ObsType, get_stddev_schedule
from data_handling.dataset import DatasetDict


class BCAgent(Agent):
    encoders: Dict[str, TrainState]
    actor: TrainState
    stddev_fn: Callable = struct.field(pytree_node=False)
    action_type: int

    @classmethod
    def create(
        cls,
        # rng: jax.Array,
        seed: int,
        lr: float,
        hidden_dim: int,
        stddev_schedule: Union[float, str],
        stddev_clip: float,
        obs_spec: Dict[str, dm_env.specs.Array],
        action_spec: dm_env.specs.Array,
        pixel_keys: List[str],
        aux_keys: List[str],
        num_queries: int,
        temporal_agg: bool,
        repr_dim: int = 256,
        use_tb: bool = False,
        augment: bool = False,
        # obs_type: ObsType = ObsType.PIXELS,
        actor_type=ActorCriticType.GPT,
        action_type=ActionType.CONTINUOUS,
    ):
        rng = jax.random.key(seed)
        rng, actor_rng = jax.random.split(rng)

        encoders = {}
        encoder_params = {}

        for aux_key in aux_keys:
            rng, aux_rng = jax.random.split(rng)
            aux_encoder = MLP(
                hidden_dims=[hidden_dim, hidden_dim],
                output_dim=repr_dim,
            )
            encoder_params[aux_key] = aux_encoder.init(
                aux_rng,
                np.random.uniform(-1.0, 1.0, (1,) + obs_spec[aux_key].shape),
            )["params"]
            encoders[aux_key] = TrainState.create(
                apply_fn=aux_encoder.apply,
                params=encoder_params[aux_key],
                tx=optax.adamw(learning_rate=lr),
            )
        if len(pixel_keys) > 0:
            rng, encoder_rng = jax.random.split(rng)
            encoder_def = ResNet18Encoder(output_dim=repr_dim)
            # encoder_def = Encoder(
            #     repr_dim=repr_dim, obs_shape=obs_spec[pixel_keys[0]].shape
            # )
            encoder_params["pixels"] = encoder_def.init(
                encoder_rng,
                np.random.uniform(0.0, 1.0, (1,) + obs_spec[pixel_keys[0]].shape),
            )["params"]
            encoders["pixels"] = TrainState.create(
                apply_fn=encoder_def.apply,
                params=encoder_params["pixels"],
                tx=optax.adam(learning_rate=lr),
            )

        num_queries = num_queries if temporal_agg else 1
        if actor_type == ActorCriticType.MLP:
            actor_def = MLPActor(
                action_dim=action_spec.shape[0] * num_queries
                if action_type == ActionType.CONTINUOUS
                else action_spec.num_values * num_queries,
                repr_dim=repr_dim,
                hidden_dim=hidden_dim,
                action_type=action_type,
            )
        else:
            actor_def = GPTActor(
                action_dim=action_spec.shape[0] * num_queries
                if action_type == ActionType.CONTINUOUS
                else action_spec.num_values * num_queries,
                hidden_dim=hidden_dim,
                repr_dim=repr_dim,
                action_type=action_type,
            )

        actor_params = actor_def.init(
            actor_rng,
            {
                key: np.random.uniform(-1.0, 1.0, (1, repr_dim))
                for key in pixel_keys + aux_keys
            },
            1.0,
        )["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=lr),
        )

        return cls(
            actor=actor,
            encoders=encoders,
            rng=rng,
            stddev_fn=get_stddev_schedule(stddev_schedule),
            action_type=int(ActionType.CONTINUOUS),
        )

    def __repr__(self):
        return "bc"

    @staticmethod
    def update_agent(agent, batch: DatasetDict, step: int):
        stddev = agent.stddev_fn(step)
        drop_key, rng = jax.random.split(agent.rng)

        def loss_fn(
            encoder_params, actor_params
        ) -> Tuple[jnp.ndarray, Dict[str, float]]:
            encoder_outputs = {}
            # LATERTODO: Profile and find the best way to do this. Might not actually matter
            for key, arr in batch.items():
                if key == "action":
                    continue
                if key.startswith("pixels"):
                    encoder_outputs[key] = agent.encoders["pixels"].apply_fn(
                        {"params": encoder_params["pixels"]}, arr, training=True
                    )
                else:
                    encoder_outputs[key] = agent.encoders[key].apply_fn(
                        {"params": encoder_params[key]}, arr, training=True
                    )
            dist = agent.actor.apply_fn(
                {"params": actor_params},
                encoder_outputs,
                stddev,
                training=True,
                rngs={"dropout": drop_key},
            )

            log_probs = dist.log_prob(
                jnp.reshape(batch["action"], (batch["action"].shape[0], -1))
            )

            loss = -jnp.mean(log_probs)
            return loss, {
                "actor_loss": loss,
                "actor_logprob": jnp.mean(log_probs),
                # LATERTODO: Return entropy
            }

        (encoder_grads, actor_grads), info = jax.grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(
            {key: agent.encoders[key].params for key in agent.encoders},
            agent.actor.params,
        )

        actor = agent.actor.apply_gradients(grads=actor_grads)

        encoders = {
            key: agent.encoders[key].apply_gradients(grads=encoder_grads[key])
            for key in agent.encoders
        }

        agent = agent.replace(actor=actor, encoders=encoders, rng=rng)

        return agent, info

    @staticmethod
    def eval_agent(agent, processed_obs: Dict[str, np.ndarray]):
        stddev = 0.0
        encoder_outputs = {}
        encoder_params = {key: agent.encoders[key].params for key in agent.encoders}

        for key, arr in processed_obs.items():
            if key == "actions":
                continue
            if key.startswith("pixels"):
                arr = jnp.expand_dims(arr, axis=0)
                encoder_outputs[key] = agent.encoders["pixels"].apply_fn(
                    {"params": encoder_params["pixels"]}, arr
                )
            else:
                arr = jnp.expand_dims(arr, axis=0)
                encoder_outputs[key] = agent.encoders[key].apply_fn(
                    {"params": encoder_params[key]}, arr
                )

        actor_params = agent.actor.params
        dist = agent.actor.apply_fn({"params": actor_params}, encoder_outputs, stddev)
        return dist.mode()

    @partial(jax.jit)
    def update(self, batch: DatasetDict, step: int):
        new_agent = self
        new_agent, info = self.update_agent(new_agent, batch, step)

        return new_agent, info

    @partial(jax.jit)
    def eval(self, processed_obs: Dict[str, np.ndarray]):
        return self.eval_agent(self, processed_obs)

    def get_save_state(self):
        agent = self
        return agent.replace(rng=jax.random.key_data(agent.rng))

    @staticmethod
    def load_state(agent_state: struct.PyTreeNode):
        return agent_state.replace(rng=jax.random.wrap_key_data(agent_state.rng))
