from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union
import dm_env
import dm_env.specs
import gym
import hydra
import jax
from jax import numpy as jnp
import omegaconf
import optax
import orbax
from flax import struct
import numpy as np
import torch
from agent.actor_critic_nets.mlp import MLPActor, MLPActorOffset, MLPCritic
from agent.actor_critic_nets.gpt import GPTActor, GPTActorOffset, GPTCritic
from agent.agent import Agent
from flax.training.train_state import TrainState

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

from flax.core.frozen_dict import FrozenDict
from rewarder import (
    cosine_distance,
    euclidean_distance,
    manhattan_distance,
    optimal_transport_plan,
)


def soft_target_update(
    critic_params: FrozenDict, target_critic_params: FrozenDict, tau: float
) -> FrozenDict:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic_params, target_critic_params
    )

    return new_target_params


class RLPDAgent(Agent):
    bc_agent: BCAgent
    actor: TrainState
    critic: TrainState
    critic_target: TrainState
    stddev_fn: Callable = struct.field(pytree_node=False)
    stddev_clip: float
    action_type: int
    critic_target_tau: float

    @classmethod
    def create(
        cls,
        bc_params: omegaconf.DictConfig,  # This is omegaconf?
        seed: int,
        lr: float,
        hidden_dim: int,
        stddev_schedule: Union[float, str],
        stddev_clip: float,
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
    ):
        # Might be a better way but right now just avoiding using the same rng as the bc agent.
        rng = jax.random.key(seed * 23)
        # rng, bc_rng = jax.random.split(rng)
        bc_agent = hydra.utils.call(
            bc_params.agent,
            seed=seed,
            obs_spec=obs_spec,
            action_spec=action_spec["action_base"],
        )

        print("Loading BC Agent from snapshot.")
        orbax_ckptr = orbax.checkpoint.PyTreeCheckpointer()
        target = {
            "agent": bc_agent.get_save_state(),
            "step": 0,
        }
        restored_state = orbax_ckptr.restore(bc_snapshot_path, item=target)
        bc_agent = bc_agent.load_state(restored_state["agent"])
        print("BC Agent loaded successfully.")

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        action_dim = (
            action_spec["action"].shape[0]
            if action_type == ActionType.CONTINUOUS
            else action_spec["action"].num_values
        )
        action_base_dim = action_spec["action_base"].shape[0]
        ac_params = dict(
            action_dim=action_dim,
            repr_dim=repr_dim,
            hidden_dim=hidden_dim,
            action_type=action_type,
        )
        if actor_type == ActorCriticType.MLP:
            actor_def = MLPActorOffset(
                **ac_params, num_obs_keys=len(aux_keys) + len(pixel_keys)
            )
        else:
            actor_def = GPTActorOffset(**ac_params)

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
                # clip gradients to prevent action collapse
                optax.clip_by_global_norm(5.0),
                optax.adamw(learning_rate=lr),
            ),
        )
        if critic_type == ActorCriticType.MLP:
            critic_def = MLPCritic(
                **ac_params,
                num_obs_keys=len(aux_keys) + len(pixel_keys),
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
                # clip gradients to prevent action collapse
                optax.clip_by_global_norm(5.0),
                optax.adamw(learning_rate=lr),
            ),
        )
        # TODO: Define critic as an ensemble instead of two separate Q functions
        critic_target = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params["params"],
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        return cls(
            bc_agent=bc_agent,
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            rng=rng,
            critic_target_tau=critic_target_tau,
            stddev_fn=get_stddev_schedule(stddev_schedule),
            stddev_clip=stddev_clip,
            action_type=int(ActionType.CONTINUOUS),
        )

    def __repr__(self):
        return "rlpd"

    @staticmethod
    def update_actor(agent, batch: DatasetDict, step: int):
        stddev = agent.stddev_fn(step)

        sample_key, rng = jax.random.split(agent.rng)
        drop_key, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            dist = agent.actor.apply_fn(
                {"params": actor_params}, batch["obs"], batch["action_base"], stddev
            )
            # TODO: Clip if required.
            offset_actions = dist.sample(seed=sample_key)
            log_prob = dist.log_prob(offset_actions)

            q1, q2 = agent.critic.apply_fn(
                {"params": agent.critic.params},
                batch["obs"],
                batch["action_base"],
                offset_actions,
                training=True,
                rngs={"dropout": drop_key},
            )
            q = jnp.minimum(q1, q2)
            actor_loss = -jnp.mean(q)

            # Add L2 and Entropy regularization
            ent_coef = 5e-3
            l2_coef = 1e-3
            ent = dist.entropy().sum(axis=-1).mean()
            entropy_reg = -ent_coef * ent
            l2_penalty = 0.0
            for p in jax.tree_leaves(actor_params):
                l2_penalty += jnp.vdot(p, p)
                l2_reg = l2_coef * l2_penalty

            actor_loss = actor_loss + entropy_reg + l2_reg

            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_logprob": log_prob.mean(),
                "actor_entropy": dist.entropy().sum(axis=-1).mean(),
                "actor_q": q.mean(),
                "rl_loss": actor_loss,
            }

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)
        agent = agent.replace(actor=actor, rng=rng)

        return agent, actor_info

    @staticmethod
    def update_critic(agent, batch: DatasetDict, step: int):
        stddev = agent.stddev_fn(step)

        sample_key, rng = jax.random.split(agent.rng)
        target_q_drop_key, rng = jax.random.split(rng)
        critic_drop_key, rng = jax.random.split(rng)

        dist = agent.actor.apply_fn(
            {"params": agent.actor.params},
            batch["obs_next"],
            batch["action_base_next"],
            stddev,
        )
        next_offset_actions = dist.sample(seed=sample_key)

        target_q1, target_q2 = agent.critic_target.apply_fn(
            {"params": agent.critic_target.params},
            batch["obs_next"],
            batch["action_base_next"],
            next_offset_actions,
            training=True,
            rngs={"dropout": target_q_drop_key},
        )

        target_V = jnp.minimum(target_q1, target_q2)
        target_q = batch["reward"] + (batch["discount"] * target_V)

        def critic_loss_fn(critic_params):
            q1, q2 = agent.critic.apply_fn(
                {"params": critic_params},
                batch["obs"],
                batch["action_base"],
                batch["action"],
                training=True,
                rngs={"dropout": critic_drop_key},
            )

            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "critic_q1": q1.mean(),
                "critic_q2": q2.mean(),
            }

        grads, critic_info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic_info["target_q"] = target_q.mean()
        critic = agent.critic.apply_gradients(grads=grads)
        critic_target_params = soft_target_update(
            critic.params, agent.critic_target.params, agent.critic_target_tau
        )
        critic_target = agent.critic_target.replace(params=critic_target_params)
        new_agent = agent.replace(critic=critic, critic_target=critic_target, rng=rng)

        return new_agent, critic_info

    @partial(jax.jit, static_argnames=("utd_ratio",))
    def update(self, batch: DatasetDict, step: int, utd_ratio: int):
        new_agent = self
        for key, arr in batch["obs"].items():
            if not key.startswith("bit"):
                if key.startswith("pixels"):
                    batch["obs"][key] = self.bc_agent.encoders["pixels"].apply_fn(
                        {"params": self.bc_agent.encoders["pixels"].params}, arr
                    )
                    batch["obs_next"][key] = self.bc_agent.encoders["pixels"].apply_fn(
                        {"params": self.bc_agent.encoders["pixels"].params},
                        batch["obs_next"][key],
                    )
                else:
                    batch["obs"][key] = self.bc_agent.encoders[key].apply_fn(
                        {"params": self.bc_agent.encoders[key].params}, arr
                    )
                    batch["obs_next"][key] = self.bc_agent.encoders[key].apply_fn(
                        {"params": self.bc_agent.encoders[key].params},
                        batch["obs_next"][key],
                    )

        mini_batches = jax.tree.map(
            lambda x: jnp.reshape(x, (utd_ratio, -1) + x.shape[1:]), batch
        )
        new_agent, critic_info = jax.lax.scan(
            lambda agent, mini_batch: self.update_critic(agent, mini_batch, step),
            init=new_agent,
            xs=mini_batches,
        )
        critic_info_final = {k: v[-1] for k, v in critic_info.items()}
        new_agent, actor_info = self.update_actor(
            new_agent, jax.tree.map(lambda x: x[-1], mini_batches), step
        )

        return new_agent, {**actor_info, **critic_info_final}

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
            {"params": self.bc_agent.actor.params}, features, stddev
        ).mode()

    @partial(jax.jit, static_argnames=("training",))
    def act_offset(self, features, action_base, stddev, training):
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, features, action_base, stddev
        )
        if not training:
            offset_action = dist.mode()
        else:
            self, key = self.split_rng(1)
            offset_action = dist.sample(seed=key)
        return offset_action

    @partial(jax.jit, static_argnames=("obs_keys", "training"))
    def act(self, obs, obs_keys, norm_stats, step, training=False):
        obs = {k: jnp.array(v) for k, v in obs.items()}

        stddev = self.stddev_fn(step)

        def pre_process(x, key):
            return (x - norm_stats["min"][key]) / (
                norm_stats["max"][key] - norm_stats["min"][key] + 1e-6
            )

        def post_process(a, key):
            return (
                a * (norm_stats["max"][key] - norm_stats["min"][key])
                + norm_stats["min"][key]
            )

        features = {}
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

        action_base = self.bc_agent.actor.apply_fn(
            {"params": self.bc_agent.actor.params}, features, stddev
        ).mode()[..., :7]

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, features, action_base, stddev
        )

        if not training:
            action = dist.mode()
        else:
            self, key = self.split_rng(1)
            action = dist.sample(seed=key)

        action_base = post_process(action_base, "action_base")
        action = post_process(action, "action")
        return action, action_base

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
            # obs is [T,8]
            target = np.array(
                [328.0, -131.0, 245.0]
            )  # np.array([340.0, -105.0, 267.0])
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
        agent = self
        agent = agent.replace(bc_agent=agent.bc_agent.get_save_state())
        return agent.replace(rng=jax.random.key_data(agent.rng))

    @staticmethod
    def load_state(agent_state: struct.PyTreeNode):
        return agent_state.replace(
            bc_agent=BCAgent.load_state(agent_state.bc_agent),
            rng=jax.random.wrap_key_data(agent_state.rng),
        )
