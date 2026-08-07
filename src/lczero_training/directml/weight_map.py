"""The correspondence between PyTorch parameters and Flax/NNX parameters.

One definition, used in both directions: the weight importer walks it to
fill a PyTorch model from Leela weights, and the tests walk it the other way
to prove the two models agree. Keeping a single source for the mapping is
what makes "every expected weight is consumed exactly once" checkable.

Layout differences handled here:

* ``nn.Linear.weight`` is ``(out, in)``; the Flax kernel is ``(in, out)``.
* ``nn.Conv2d.weight`` is ``(out, in/groups, kh, kw)``; the Flax depthwise
  kernel is ``(kh, kw, in/groups, out)``.

Everything else -- norms, gates, KDA decay parameters -- matches elementwise.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


class Layout(enum.Enum):
    """How a PyTorch array relates to its Flax counterpart."""

    DIRECT = "direct"
    LINEAR = "linear"  # transpose
    CONV = "conv"  # (out, in, kh, kw) <-> (kh, kw, in, out)


@dataclasses.dataclass(frozen=True)
class ParamPair:
    name: str
    torch_param: torch.nn.Parameter
    jax_param: Any
    layout: Layout

    def jax_to_numpy(self) -> np.ndarray:
        """The Flax value, in PyTorch layout."""
        value = np.asarray(self.jax_param.value)
        if self.layout is Layout.LINEAR:
            return value.T
        if self.layout is Layout.CONV:
            return value.transpose(3, 2, 0, 1)
        return value

    def torch_to_numpy(self) -> np.ndarray:
        """The PyTorch value, in Flax layout."""
        value = self.torch_param.detach().cpu().numpy()
        if self.layout is Layout.LINEAR:
            return value.T
        if self.layout is Layout.CONV:
            return value.transpose(2, 3, 1, 0)
        return value


def _linear(name: str, torch_linear, jax_linear) -> Iterator[ParamPair]:
    yield ParamPair(
        f"{name}.weight", torch_linear.weight, jax_linear.kernel, Layout.LINEAR
    )
    if torch_linear.bias is not None:
        yield ParamPair(
            f"{name}.bias", torch_linear.bias, jax_linear.bias, Layout.DIRECT
        )


def _norm(name: str, torch_norm, jax_norm) -> Iterator[ParamPair]:
    yield ParamPair(
        f"{name}.scale", torch_norm.scale, jax_norm.scale, Layout.DIRECT
    )
    yield ParamPair(
        f"{name}.bias", torch_norm.bias, jax_norm.bias, Layout.DIRECT
    )


def _ffn(name: str, torch_ffn, jax_ffn) -> Iterator[ParamPair]:
    yield from _linear(f"{name}.linear1", torch_ffn.linear1, jax_ffn.linear1)
    yield from _linear(f"{name}.linear2", torch_ffn.linear2, jax_ffn.linear2)


def _kda(name: str, torch_mixer, jax_mixer) -> Iterator[ParamPair]:
    for attr in ("q", "k", "v", "decay_a", "decay_b", "beta", "output_dense"):
        yield from _linear(
            f"{name}.{attr}",
            getattr(torch_mixer, attr),
            getattr(jax_mixer, attr),
        )
    if torch_mixer.output_gate:
        yield from _linear(
            f"{name}.gate_a", torch_mixer.gate_a, jax_mixer.gate_a
        )
        yield from _linear(
            f"{name}.gate_b", torch_mixer.gate_b, jax_mixer.gate_b
        )
    if torch_mixer.local_conv is not None:
        yield ParamPair(
            f"{name}.local_conv.weight",
            torch_mixer.local_conv.conv.weight,
            jax_mixer.local_conv.conv.kernel,
            Layout.CONV,
        )
        yield ParamPair(
            f"{name}.local_conv.bias",
            torch_mixer.local_conv.conv.bias,
            jax_mixer.local_conv.conv.bias,
            Layout.DIRECT,
        )
    if torch_mixer.rms_norm is not None:
        yield ParamPair(
            f"{name}.rms_norm.scale",
            torch_mixer.rms_norm.scale,
            jax_mixer.rms_norm_gammas,
            Layout.DIRECT,
        )
    yield ParamPair(
        f"{name}.log_decay.a_log",
        torch_mixer.log_decay.a_log,
        jax_mixer.log_decay.a_log,
        Layout.DIRECT,
    )
    yield ParamPair(
        f"{name}.log_decay.dt_bias",
        torch_mixer.log_decay.dt_bias,
        jax_mixer.log_decay.dt_bias,
        Layout.DIRECT,
    )


def _mha(name: str, torch_mha, jax_mha) -> Iterator[ParamPair]:
    for attr in ("q", "k", "v", "output_dense"):
        yield from _linear(
            f"{name}.{attr}", getattr(torch_mha, attr), getattr(jax_mha, attr)
        )
    if torch_mha.smolgen is not None:
        smol, jax_smol = torch_mha.smolgen, jax_mha.smolgen
        yield from _linear(
            f"{name}.smolgen.compress", smol.compress, jax_smol.compress
        )
        yield from _linear(
            f"{name}.smolgen.dense1", smol.dense1, jax_smol.dense1
        )
        yield from _norm(f"{name}.smolgen.ln1", smol.ln1, jax_smol.ln1)
        yield from _linear(
            f"{name}.smolgen.dense2", smol.dense2, jax_smol.dense2
        )
        yield from _norm(f"{name}.smolgen.ln2", smol.ln2, jax_smol.ln2)


def _jax_shared_smolgen_dense(jax_model) -> Any | None:
    """The tower-wide Smolgen generator, or None when unused.

    It is one object referenced by every MHA block, so it must be visited
    exactly once however many MHA blocks there are.
    """
    for block in jax_model.encoders.encoders.layers:
        if not block.is_kda and block.mha.smolgen is not None:
            return block.mha.smolgen.weight_gen_dense
    return None


def iter_param_pairs(torch_model, jax_model) -> Iterator[ParamPair]:
    """Every trainable array, paired between the two models."""
    torch_embed, jax_embed = torch_model.embedding, jax_model.embedding
    yield from _linear(
        "embedding.preprocess", torch_embed.preprocess, jax_embed.preprocess
    )
    yield from _linear(
        "embedding.embedding", torch_embed.embedding, jax_embed.embedding
    )
    yield from _norm("embedding.norm", torch_embed.norm, jax_embed.norm)
    yield ParamPair(
        "embedding.ma_gating.mult",
        torch_embed.ma_gating.mult_gate.gate,
        jax_embed.ma_gating.mult_gate.gate,
        Layout.DIRECT,
    )
    yield ParamPair(
        "embedding.ma_gating.add",
        torch_embed.ma_gating.add_gate.gate,
        jax_embed.ma_gating.add_gate.gate,
        Layout.DIRECT,
    )
    yield from _ffn("embedding.ffn", torch_embed.ffn, jax_embed.ffn)
    yield from _norm(
        "embedding.out_norm", torch_embed.out_norm, jax_embed.out_norm
    )

    shared = torch_model.encoders.smolgen_shared_gen_dense
    if shared is not None:
        jax_shared = _jax_shared_smolgen_dense(jax_model)
        assert jax_shared is not None, (
            "torch tower has a shared Smolgen dense but the JAX tower does not"
        )
        yield from _linear("encoders.smolgen_shared", shared, jax_shared)

    torch_blocks = list(torch_model.encoders.encoders)
    jax_blocks = list(jax_model.encoders.encoders.layers)
    assert len(torch_blocks) == len(jax_blocks), "encoder depth mismatch"
    for index, (torch_block, jax_block) in enumerate(
        zip(torch_blocks, jax_blocks)
    ):
        assert torch_block.is_kda == jax_block.is_kda, (
            f"encoder block {index} mixer type mismatch"
        )
        name = f"encoders.{index}"
        if torch_block.is_kda:
            yield from _kda(f"{name}.mixer", torch_block.mixer, jax_block.mixer)
        else:
            yield from _mha(f"{name}.mha", torch_block.mha, jax_block.mha)
        yield from _norm(f"{name}.ln1", torch_block.ln1, jax_block.ln1)
        yield from _ffn(f"{name}.ffn", torch_block.ffn, jax_block.ffn)
        yield from _norm(f"{name}.ln2", torch_block.ln2, jax_block.ln2)

    if torch_model.policy_embedding_shared is not None:
        yield from _linear(
            "policy_embedding_shared",
            torch_model.policy_embedding_shared,
            jax_model.policy_embedding_shared,
        )
    for head_name, head in torch_model.policy_heads.items():
        jax_head = jax_model.policy_heads[head_name]
        name = f"policy_heads.{head_name}"
        # With a shared embedding, `tokens` is that same shared Linear --
        # already yielded above, so do not yield it again per head.
        if torch_model.policy_embedding_shared is None:
            yield from _linear(f"{name}.tokens", head.tokens, jax_head.tokens)
        yield from _linear(f"{name}.q", head.q, jax_head.q)
        yield from _linear(f"{name}.k", head.k, jax_head.k)
        yield from _linear(
            f"{name}.promotion_dense",
            head.promotion_dense,
            jax_head.promotion_dense,
        )

    for head_name, head in torch_model.value_heads.items():
        jax_head = jax_model.value_heads[head_name]
        name = f"value_heads.{head_name}"
        yield from _linear(f"{name}.embed", head.embed, jax_head.embed)
        yield from _linear(f"{name}.dense1", head.dense1, jax_head.dense1)
        yield from _linear(f"{name}.wdl", head.wdl, jax_head.wdl)
        if head.error is not None:
            yield from _linear(f"{name}.error", head.error, jax_head.error)
        if head.categorical is not None:
            yield from _linear(
                f"{name}.categorical", head.categorical, jax_head.categorical
            )

    for head_name, head in torch_model.movesleft_heads.items():
        jax_head = jax_model.movesleft_heads[head_name]
        name = f"movesleft_heads.{head_name}"
        yield from _linear(f"{name}.embed", head.embed, jax_head.embed)
        yield from _linear(f"{name}.dense1", head.dense1, jax_head.dense1)
        yield from _linear(f"{name}.out", head.out, jax_head.out)


def copy_jax_to_torch(torch_model, jax_model) -> int:
    """Fill the PyTorch model from the Flax one. Returns the pair count."""
    count = 0
    with torch.no_grad():
        for pair in iter_param_pairs(torch_model, jax_model):
            value = pair.jax_to_numpy()
            if tuple(value.shape) != tuple(pair.torch_param.shape):
                raise ValueError(
                    f"{pair.name}: Leela weight has shape {value.shape}, "
                    f"model expects {tuple(pair.torch_param.shape)}"
                )
            # .copy() not .ascontiguousarray(): jax returns read-only arrays,
            # and torch.from_numpy on one of those warns about undefined
            # behaviour on write.
            pair.torch_param.copy_(
                torch.from_numpy(np.ascontiguousarray(value).copy())
            )
            count += 1
    return count


def copy_torch_to_jax(torch_model, jax_model) -> int:
    """Fill the Flax model from the PyTorch one. Returns the pair count."""
    import jax.numpy as jnp

    count = 0
    for pair in iter_param_pairs(torch_model, jax_model):
        pair.jax_param.value = jnp.asarray(pair.torch_to_numpy())
        count += 1
    return count
