import math
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze
from flax.traverse_util import path_aware_map

import numpy as np
import optax


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_dim: int = 256
    output_dim: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(3 * config.n_embd)
        # output projection
        self.c_proj = nn.Dense(config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x: jax.Array, training: bool = False):
        B, T, C = x.shape

        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2)

        mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1]))).reshape(1, 1, T, T)

        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = jnp.where(mask == 0, float("-inf"), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not training)

        y = att @ v
        y = y.swapaxes(1, 2).reshape(B, T, C)

        y = self.resid_dropout(self.c_proj(y), deterministic=not training)
        return y


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.c_fc = nn.Dense(config.n_embd * 4)
        self.c_proj = nn.Dense(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: jax.Array, training: bool = False):
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not training)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.ln1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(config)

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.input_dim is not None
        assert config.output_dim is not None
        assert config.block_size is not None

        self.wte = nn.Dense(config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5)

        self.lm_head = nn.Dense(config.output_dim, use_bias=False)
        # TODO: Check weight initialization

    def __call__(self, x: jax.Array, *, training: bool = False) -> jax.Array:
        b, t, d = x.shape
        assert (
            t <= self.config.block_size
        ), "Cannot forward, model block size is exhausted."
        pos = jnp.arange(0, t, dtype=jnp.int32)[None, :]

        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, deterministic=not training)
        for block in self.h:
            x = block(x, training=training)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        return logits

    def crop_block_size(self, block_size):
        assert (
            block_size <= self.config.block_size
        ), "Cannot crop block size, new block size is larger than original."
        self.config.block_size = block_size

        def crop_weights(path: Tuple[str, ...], w):
            if path[-2:] == ("wpe", "embedding"):
                return w[:block_size]
            return w

        return freeze(path_aware_map(crop_weights, self.params))

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        # TODO: Figure out what exactly is going on here
        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate,
                b1=betas[0],
                b2=betas[1],
                weight_decay=decay,
            )

        def partition_fn(path: Tuple[str, ...], x) -> str:
            # TODO: Check if this is accounting for layernorm as well
            if path[-1] in ("bias", "scale", "embedding"):
                return "no_decay"
            elif path[-1] in ("kernel",):
                return "decay"
            else:
                raise ValueError(f"Unexpected parameter: {path}")

        partition_optimizers = {
            "decay": get_optimizer(weight_decay),
            "no_decay": get_optimizer(0.0),
        }
        param_partitions = freeze(path_aware_map(partition_fn, params))
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        return tx
