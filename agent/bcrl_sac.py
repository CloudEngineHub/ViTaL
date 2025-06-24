from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union
import dm_env
import dm_env.specs
import gym
import hydra
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
import omegaconf
import optax
import orbax
from flax import struct
import numpy as np
import torch
import distrax
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import time
from agent.actor_critic_nets.mlp import MLPActor, MLPActorOffsetSAC, MLPCriticSAC
from agent.actor_critic_nets.gpt import GPTActor, GPTActorOffset, GPTCritic
from agent.agent import Agent
from agent.bc import BCAgent
from agent.networks.encoder import Encoder
from agent.networks.mlp import MLP
from agent.networks.resnet_encoder import ResNet18Encoder
from utils import (
    ActionType,
    ActorCriticType,
    ObsType,
    get_stddev_schedule,
)
from data_handling.dataset import DatasetDict
from rewarder import (
    cosine_distance,
    euclidean_distance,
    manhattan_distance,
    optimal_transport_plan,
)


class TemperatureLagrange(nn.Module):
    """Temperature Lagrange multiplier for entropy regularization"""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_temperature = self.param(
            "log_temperature", lambda key: jnp.log(self.init_value) * jnp.ones(())
        )
        return jnp.exp(log_temperature)


def soft_target_update(
    critic_params: FrozenDict, target_critic_params: FrozenDict, tau: float
) -> FrozenDict:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic_params, target_critic_params
    )
    return new_target_params


class BCRLSACAgent(Agent):
    bc_agent: BCAgent
    actor: TrainState
    critic: TrainState
    critic_target: TrainState
    temperature: TrainState
    action_type: int
    critic_target_tau: float
    target_entropy: float
    backup_entropy: bool
    critic_ensemble_size: int
    critic_subsample_size: Optional[int]

    @classmethod
    def create(
        cls,
        bc_params: omegaconf.DictConfig,
        seed: int,
        lr: float,
        hidden_dim: int,
        obs_spec: Dict[str, dm_env.specs.Array],
        action_spec: Dict[str, dm_env.specs.Array],
        pixel_keys: List[str],
        aux_keys: List[str],
        critic_target_tau: float,
        repr_dim: int = 256,
        actor_type=ActorCriticType.MLP,
        critic_type=ActorCriticType.MLP,
        action_type=ActionType.CONTINUOUS,
        bc_snapshot_path: Optional[str] = None,
        use_layer_norm: bool = False,
        critic_dropout_rate: Optional[float] = None,
        temperature_lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_ensemble_size: int = 2,
        critic_subsample_size: int = 2,
        temperature_init: float = 1.0,
    ):
        # Might be a better way but right now just avoiding using the same rng as the bc agent.
        rng = jax.random.key(seed * 23)
        bc_agent = hydra.utils.call(
            bc_params.agent,
            seed=seed,
            obs_spec=obs_spec,
            action_spec=action_spec["action_base"],
        )

        if bc_snapshot_path:
            print("Loading BC Agent from snapshot.")
            orbax_ckptr = orbax.checkpoint.PyTreeCheckpointer()
            target = {
                "agent": bc_agent.get_save_state(),
                "step": 0,
            }
            restored_state = orbax_ckptr.restore(bc_snapshot_path, item=target)
            bc_agent = bc_agent.load_state(restored_state["agent"])
            print("BC Agent loaded successfully.")

        rng, actor_rng, critic_rng, temp_rng = jax.random.split(rng, 4)

        action_dim = (
            action_spec["action"].shape[0]
            if action_type == ActionType.CONTINUOUS
            else action_spec["action"].num_values
        )
        action_base_dim = action_spec["action_base"].shape[0]

        # Set target entropy
        if target_entropy is None:
            target_entropy = -action_dim / 2

        ac_params = dict(
            action_dim=action_dim,
            repr_dim=repr_dim,
            hidden_dim=hidden_dim,
            action_type=action_type,
            num_obs_keys=len(aux_keys) + len(pixel_keys),
        )

        if actor_type == ActorCriticType.MLP:
            actor_def = MLPActorOffsetSAC(
                **ac_params,
                std_parameterization="exp",
                std_min=1e-5,
                std_max=1.0,
                tanh_squash_distribution=True,
                init_final=0.01,
            )
        else:
            actor_def = GPTActorOffset(**ac_params)  # not implemented for SAC

        actor_params = actor_def.init(
            actor_rng,
            {
                key: np.random.uniform(-1.0, 1.0, (1, repr_dim))
                for key in pixel_keys + aux_keys
            },
            np.random.uniform(-1.0, 1.0, (1, action_base_dim)),
            1.0,
        )["params"]

        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.chain(
                optax.clip_by_global_norm(5.0),
                optax.adamw(learning_rate=lr, weight_decay=5e-5),
            ),
        )

        if critic_type == ActorCriticType.MLP:
            critic_def = MLPCriticSAC(
                **ac_params,
                ensemble_size=critic_ensemble_size,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=use_layer_norm,
            )
        elif critic_type == ActorCriticType.GPT:
            critic_def = GPTCritic(**ac_params)

        critic_params = critic_def.init(
            critic_rng,
            {
                key: np.random.uniform(-1.0, 1.0, (1, repr_dim))
                for key in pixel_keys + aux_keys
            },
            np.random.uniform(-1.0, 1.0, (1, action_base_dim)),
            np.random.uniform(-1.0, 1.0, (1, action_dim)),
        )

        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params["params"],
            tx=optax.chain(
                optax.clip_by_global_norm(5.0),
                optax.adamw(learning_rate=lr),
            ),
        )

        critic_target = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params["params"],
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        # Initialize temperature(alpha)
        temperature_def = TemperatureLagrange(init_value=temperature_init)
        temperature_params = temperature_def.init(temp_rng)
        temperature = TrainState.create(
            apply_fn=temperature_def.apply,
            params=temperature_params["params"],
            tx=optax.adamw(learning_rate=temperature_lr),
        )

        return cls(
            bc_agent=bc_agent,
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            temperature=temperature,
            rng=rng,
            critic_target_tau=critic_target_tau,
            action_type=int(ActionType.CONTINUOUS),
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
        )

    def __repr__(self):
        return "bcrl_sac"

    @staticmethod
    @jax.jit
    def update_actor(agent, batch: DatasetDict, step: int):
        sample_key, rng = jax.random.split(agent.rng)
        drop_key, rng = jax.random.split(rng)

        temperature = agent.temperature.apply_fn({"params": agent.temperature.params})

        def actor_loss_fn(actor_params):
            dist = agent.actor.apply_fn(
                {"params": actor_params},
                batch["obs"],
                batch["action_base"],
                temperature,
                training=True,
            )

            offset_actions = dist.sample(seed=sample_key)
            log_prob = dist.log_prob(offset_actions)

            log_prob = jnp.where(
                log_prob.ndim > 1, jnp.sum(log_prob, axis=-1), log_prob
            )

            # Shape: (ensemble_size, batch_size)
            qs = agent.critic.apply_fn(
                {"params": agent.critic.params},
                batch["obs"],
                batch["action_base"],
                offset_actions,
                training=True,
                rngs={"dropout": drop_key},
            )

            qs = qs.squeeze() if qs.ndim > 2 else qs
            q_min = jnp.min(qs, axis=0)

            # SAC actor objective
            actor_objective = q_min - temperature * log_prob
            actor_loss = -jnp.mean(actor_objective)

            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_logprob": log_prob.mean(),
                "actor_entropy": -log_prob.mean(),
                "actor_q": q_min.mean(),
                "temperature": temperature,
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)
        new_agent = agent.replace(actor=actor, rng=rng)

        return new_agent, actor_info

    @staticmethod
    @jax.jit
    def update_critic(agent, batch: DatasetDict, step: int):
        sample_key, rng = jax.random.split(agent.rng)
        target_q_drop_key, rng = jax.random.split(rng)
        critic_drop_key, rng = jax.random.split(rng)

        temperature = agent.temperature.apply_fn({"params": agent.temperature.params})

        dist = agent.actor.apply_fn(
            {"params": agent.actor.params},
            batch["obs_next"],
            batch["action_base_next"],
            temperature,
            training=False,
        )
        next_offset_actions = dist.sample(seed=sample_key)
        next_log_probs = dist.log_prob(next_offset_actions)

        next_log_probs = jnp.where(
            next_log_probs.ndim > 1, jnp.sum(next_log_probs, axis=-1), next_log_probs
        )

        # Shape: (ensemble_size, batch_size)
        target_qs = agent.critic_target.apply_fn(
            {"params": agent.critic_target.params},
            batch["obs_next"],
            batch["action_base_next"],
            next_offset_actions,
            training=True,
            rngs={"dropout": target_q_drop_key},
        )

        target_qs = target_qs.squeeze() if target_qs.ndim > 2 else target_qs
        target_q_min = jnp.min(target_qs, axis=0)
        target_V = jnp.where(
            agent.backup_entropy,
            target_q_min - temperature * next_log_probs,
            target_q_min,
        )

        discount = batch.get("discount", jnp.ones_like(batch["reward"]))
        target_q = batch["reward"] + discount * target_V

        def critic_loss_fn(critic_params):
            qs = agent.critic.apply_fn(
                {"params": critic_params},
                batch["obs"],
                batch["action_base"],
                batch["action"],
                training=True,
                rngs={"dropout": critic_drop_key},
            )

            qs = qs.squeeze() if qs.ndim > 2 else qs
            target_q_expanded = jnp.expand_dims(target_q, axis=0)

            critic_loss = jnp.mean((qs - target_q_expanded) ** 2)

            metric = {
                "critic_loss": critic_loss,
                "critic_q_mean": jnp.mean(qs),
                "critic_q_std": jnp.std(qs),
                "critic_q_min": jnp.min(qs),
                "critic_q_max": jnp.max(qs),
                "target_q": target_q.mean(),
            }

            return critic_loss, metric

        grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        critic_target_params = soft_target_update(
            critic.params, agent.critic_target.params, agent.critic_target_tau
        )
        critic_target = agent.critic_target.replace(params=critic_target_params)

        new_agent = agent.replace(critic=critic, critic_target=critic_target, rng=rng)

        return new_agent, critic_info

    @staticmethod
    @jax.jit
    def update_temperature(agent, batch: DatasetDict, step: int):
        sample_key, rng = jax.random.split(agent.rng)

        def temperature_loss_fn(temperature_params):
            temperature = agent.temperature.apply_fn({"params": temperature_params})

            dist = agent.actor.apply_fn(
                {"params": agent.actor.params},
                batch["obs"],
                batch["action_base"],
                temperature,
                training=False,
            )
            actions = dist.sample(seed=sample_key)
            log_probs = dist.log_prob(actions)

            log_probs = jnp.where(
                log_probs.ndim > 1, jnp.sum(log_probs, axis=-1), log_probs
            )

            entropy = -log_probs.mean()

            temperature_loss = temperature * (entropy - agent.target_entropy)

            return temperature_loss, {
                "temperature_loss": temperature_loss,
                "temperature": temperature,
                "entropy": entropy,
                "target_entropy": agent.target_entropy,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(
            agent.temperature.params
        )
        temperature = agent.temperature.apply_gradients(grads=grads)

        new_agent = agent.replace(temperature=temperature, rng=rng)
        return new_agent, temp_info

    @partial(jax.jit, static_argnames=("utd_ratio",))
    def update(
        self,
        batch: Dict[str, jnp.ndarray],
        step: int,
        utd_ratio: int,
    ):
        new_agent = self

        def encode_obs_dict(obs_dict, encoders):
            encoded = {}
            for key, arr in obs_dict.items():
                if key.startswith("bit"):
                    encoded[key] = arr
                elif key.startswith("pixels"):
                    encoded[key] = encoders["pixels"].apply_fn(
                        {"params": encoders["pixels"].params}, arr
                    )
                else:
                    encoded[key] = encoders[key].apply_fn(
                        {"params": encoders[key].params}, arr
                    )
            return encoded

        encoded_batch = dict(batch)
        encoded_batch["obs"] = encode_obs_dict(batch["obs"], self.bc_agent.encoders)
        encoded_batch["obs_next"] = encode_obs_dict(
            batch["obs_next"], self.bc_agent.encoders
        )

        def reshape_for_utd(x):
            return jnp.reshape(x, (utd_ratio, -1) + x.shape[1:])

        mini_batches = jax.tree_map(reshape_for_utd, encoded_batch)

        new_agent, critic_infos = jax.lax.scan(
            lambda agent, mb: self.update_critic(agent, mb, step),
            init=new_agent,
            xs=mini_batches,
        )

        critic_info = jax.tree_map(lambda x: x[-1], critic_infos)
        new_agent, actor_info = self.update_actor(new_agent, encoded_batch, step)

        new_agent, temp_info = self.update_temperature(new_agent, encoded_batch, step)

        return new_agent, {**actor_info, **critic_info, **temp_info}

    @partial(jax.jit, static_argnames=("key"))
    def pre_process(self, x, norm_stats, key):
        return (x - norm_stats["min"][key]) / (
            norm_stats["max"][key] - norm_stats["min"][key] + 1e-6
        )

    @partial(jax.jit, static_argnames=("key"))
    def post_process(self, a, norm_stats, key):
        return (
            a * (norm_stats["max"][key] - norm_stats["min"][key])
            + norm_stats["min"][key]
        )

    @partial(jax.jit, static_argnames=("obs_keys",))
    def extract_features(self, obs, obs_keys, norm_stats):
        def pre_process(x, key):
            return (x - norm_stats["min"][key]) / (
                norm_stats["max"][key] - norm_stats["min"][key]
            )

        features = {}
        obs = {k: jnp.array(v) for k, v in obs.items()}
        for key in obs_keys:
            if key.startswith("pixels"):
                proc_obs = jnp.expand_dims(obs[key] / 255.0, axis=0)
                features[key] = self.bc_agent.encoders["pixels"].apply_fn(
                    {"params": self.bc_agent.encoders["pixels"].params}, proc_obs
                )
            else:
                proc_obs = jnp.expand_dims(pre_process(obs[key], key), axis=0)
                features[key] = self.bc_agent.encoders[key].apply_fn(
                    {"params": self.bc_agent.encoders[key].params}, proc_obs
                )
        return features

    @jax.jit
    def act_base(self, features, stddev):
        return self.bc_agent.actor.apply_fn(
            {"params": self.bc_agent.actor.params}, features, std=0.01
        ).mode()

    @partial(jax.jit, static_argnames=("training",))
    def act_offset(self, features, action_base, temperature, training, rng):
        dist = self.actor.apply_fn(
            {"params": self.actor.params},
            features,
            action_base,
            temperature,
            training=training,
        )

        offset_action = dist.sample(seed=rng)
        return offset_action

    @partial(jax.jit, static_argnames=("key"))
    def encode_obs(self, obs, key, stats):
        if key.startswith("pixels"):
            obs = obs / 255.0
            obs = self.bc_agent.encoders["pixels"].apply_fn(
                {"params": self.bc_agent.encoders["pixels"].params}, obs
            )
        elif key.startswith("sensor"):
            obs = (obs - stats["min"][key]) / (stats["max"][key] - stats["min"][key])
            obs = self.bc_agent.encoders[key].apply_fn(
                {"params": self.bc_agent.encoders[key].params}, obs
            )
        else:
            raise NotImplementedError("Observation type not supported.")
        return obs

    def multi_rewarder(
        self,
        single_obs: np.ndarray,
        expert_demos: List[Dict[str, np.ndarray]],
        key,
        stats,
        sinkhorn_rew_scale_dict,
        reward_type,
        use_raw,
    ):
        scores_list = []
        irl_rewards_list = []

        single_obs = np.array(single_obs)

        if key == "proprioceptive":
            single_obs = single_obs[:, :3]  # Only x,y,z proprio
            target = np.array([311.0, -80.0, 265.0])
            diff = np.abs(single_obs - target) / 1000  # shape [T, 3]

            weighted_diff = diff[:, 0] * 10 + diff[:, 1] * 10 + diff[:, 2] * 10
            reward_list = -weighted_diff

            return reward_list

        if not use_raw:
            single_obs = self.encode_obs(single_obs, key, stats)

        for demo in expert_demos:
            if key == "proprioceptive":
                demo_data = np.concatenate(
                    (
                        demo["cartesian_states"],
                        np.zeros((len(demo["cartesian_states"]), 2)),
                    ),
                    axis=1,
                )
                demo_data = demo_data[:, :3]
            else:
                demo_data = demo[key]

            if not use_raw:
                demo_data = self.encode_obs(demo_data, key, stats)

            # OT reward computation
            if reward_type == "sinkhorn_cosine":
                cost_matrix = cosine_distance(single_obs, demo_data)
                transport_plan = optimal_transport_plan(
                    single_obs, demo_data, cost_matrix, method="sinkhorn", niter=500
                )

            elif reward_type == "sinkhorn_euclidean":
                cost_matrix = euclidean_distance(single_obs, demo_data)

                transport_plan = optimal_transport_plan(
                    single_obs,
                    demo_data,
                    cost_matrix,
                    method="sinkhorn",
                    niter=500,
                    epsilon=1,
                )

            elif reward_type == "sinkhorn_manhattan":
                cost_matrix = manhattan_distance(single_obs, demo_data)
                transport_plan = optimal_transport_plan(
                    single_obs,
                    demo_data,
                    cost_matrix,
                    method="sinkhorn",
                    niter=500,
                    epsilon=1,
                )

            irl_rewards = -sinkhorn_rew_scale_dict[key] * np.diag(
                np.dot(transport_plan, cost_matrix.T)
            )

            scores_list.append(np.sum(irl_rewards))
            irl_rewards_list.append(irl_rewards)

        threshold = 1e-4
        valid_scores = [
            (score, idx)
            for idx, score in enumerate(scores_list)
            if abs(score) > threshold
        ]
        if valid_scores:
            best_idx = max(valid_scores, key=lambda x: x[0])[1]
        else:
            best_idx = np.argmax(scores_list)

        return irl_rewards_list[best_idx]

    def get_save_state(self):
        return {
            "bc_agent": self.bc_agent.get_save_state(),
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "temperature": self.temperature,
            "rng": jax.random.key_data(self.rng),
            "critic_target_tau": self.critic_target_tau,
            "target_entropy": self.target_entropy,
            "backup_entropy": self.backup_entropy,
            "action_type": self.action_type,
            "critic_ensemble_size": self.critic_ensemble_size,
            "critic_subsample_size": self.critic_subsample_size,
        }

    @staticmethod
    def load_state(agent_state: struct.PyTreeNode):
        loaded_agent = agent_state.copy()
        loaded_agent["bc_agent"] = BCAgent.load_state(agent_state["bc_agent"])
        loaded_agent["rng"] = jax.random.wrap_key_data(agent_state["rng"])
        return BCRLSACAgent(**loaded_agent)
