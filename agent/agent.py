from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

# from jax.random import KeyArray
import numpy as np
from flax import struct
from flax.training.train_state import TrainState


@partial(jax.jit, static_argnames=["enc_apply_fns", "actor_apply_fn"])
def _sample_actions(
    rng,
    enc_apply_fns,
    actor_apply_fn,
    enc_params,
    actor_params,
    observations: Dict[str, np.ndarray],
) -> np.ndarray:
    key, rng = jax.random.split(rng)
    encoder_outputs = {}
    for k in observations:
        if k.startswith("pixels"):
            encoder_outputs[k] = enc_apply_fns["pixels"](
                {"params": enc_params["pixels"]}, observations[k][None, ...]
            )
        else:
            encoder_outputs[k] = enc_apply_fns[k](
                {"params": enc_params[k]}, observations[k][None, ...]
            )
    dist = actor_apply_fn({"params": actor_params}, encoder_outputs)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames=["enc_apply_fns", "actor_apply_fn"])
def _eval_actions(
    enc_apply_fns,
    actor_apply_fn,
    enc_params,
    actor_params,
    observations: Dict[str, np.ndarray],
) -> np.ndarray:
    encoder_outputs = {}
    for k in observations:
        if k.startswith("pixels"):
            encoder_outputs[k] = enc_apply_fns["pixels"](
                {"params": enc_params["pixels"]}, observations[k][None, ...]
            )
        else:
            encoder_outputs[k] = enc_apply_fns[k](
                {"params": enc_params[k]}, observations[k][None, ...]
            )
    dist = actor_apply_fn({"params": actor_params}, encoder_outputs)
    return dist.mode()


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: jax.Array

    def eval_actions(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        # Maybe move this to a separate function?
        enc_apply_fns = {}
        enc_params = {}
        for k, enc in self.encoders.items():
            enc_apply_fns[k] = enc.apply_fn
            enc_params[k] = enc

        actions = _eval_actions(
            enc_apply_fns,
            self.actor.apply_fn,
            enc_params,
            self.actor.params,
            observations,
        )
        return np.asarray(actions)

    def split_rng(self, n: int):
        new_rng, *rngs = jax.random.split(self.rng, n + 1)
        return self.replace(rng=new_rng), *rngs

    def sample_actions(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        enc_apply_fns = {}
        enc_params = {}
        for k, enc in self.encoders.items():
            enc_apply_fns[k] = enc.apply_fn
            enc_params[k] = enc
        actions, new_rng = _sample_actions(
            self.rng,
            enc_apply_fns,
            self.actor.apply_fn,
            enc_params,
            self.actor.params,
            observations,
        )
        return np.asarray(actions), self.replace(rng=new_rng)
