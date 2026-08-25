import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from proto import model_config_pb2

from .policy_map import POLICY_MAP
from .utils import get_activation


class PolicyHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        config: model_config_pb2.PolicyHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        shared_embedding: Optional[nnx.Linear] = None,
        *,
        rngs: nnx.Rngs,
    ):
        assert (shared_embedding is not None) != config.HasField(
            "embedding_size"
        )
        self.activation = defaults.activation
        if shared_embedding is not None:
            self.tokens = shared_embedding
            embedding_size = shared_embedding.out_features
        else:
            self.tokens = nnx.Linear(
                in_features=in_features,
                out_features=config.embedding_size,
                rngs=rngs,
            )
            embedding_size = config.embedding_size

        self.q = nnx.Linear(
            in_features=embedding_size,
            out_features=config.d_model,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        self.k = nnx.Linear(
            in_features=embedding_size,
            out_features=config.d_model,
            use_bias=config.use_bias,
            rngs=rngs,
        )

        self.dk = math.sqrt(config.d_model)
        self.promotion_dense = nnx.Linear(
            in_features=config.d_model,
            out_features=4,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.tokens(x)
        x = get_activation(self.activation)(x)

        q = self.q(x)
        k = self.k(x)
        qk = jnp.einsum("qd,kd->qk", q, k)

        promotion_keys = k[-8:, :]
        promotion_offsets = self.promotion_dense(promotion_keys)
        promotion_offsets = promotion_offsets.transpose((1, 0)) * self.dk
        # knight offset is added to the other three
        promotion_offsets = promotion_offsets[:3, :] + promotion_offsets[3:4, :]

        n_promo_logits = qk[-16:-8, -8:]
        q_promo_logits = jnp.expand_dims(
            n_promo_logits + promotion_offsets[0:1, :], axis=-1
        )
        r_promo_logits = jnp.expand_dims(
            n_promo_logits + promotion_offsets[1:2, :], axis=-1
        )
        b_promo_logits = jnp.expand_dims(
            n_promo_logits + promotion_offsets[2:3, :], axis=-1
        )
        promotion_logits = jnp.concatenate(
            [q_promo_logits, r_promo_logits, b_promo_logits], axis=-1
        )
        policy_attn_logits = qk / self.dk
        promotion_logits = promotion_logits.reshape((8, 24)) / self.dk

        logits = jnp.concatenate(
            [policy_attn_logits.flatten(), promotion_logits.flatten()], axis=-1
        )

        return logits[_policy_map]


# The table lives in policy_map.py so the PyTorch/DirectML port shares one
# copy; test_heads.py asserts the two stay identical.
_policy_map = jnp.array(POLICY_MAP, jnp.int32)
