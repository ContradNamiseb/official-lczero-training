"""Is the factored fast path less accurate than the pairwise slow path?

Both are exact in real arithmetic. The factored form evaluates
exp(cum[i]) * exp(-cum[j]) -- two numbers that can be 1e-35 and 1e+34 --
where the pairwise form evaluates exp(cum[i] - cum[j]) directly from a
small difference. Catastrophic cancellation says the factored form should
lose ground as the gate saturates. Measured against a float64 sequential
reference, which is the ground truth for both.
"""

import numpy as np
import torch

import lczero_training.directml.kda as kda_mod
from lczero_training.directml.kda import KDA_LOG_DECAY_FLOOR, kda_recurrence


def reference(query, key, value, log_decay, beta):
    dt = np.float64
    query = query.astype(dt)
    key = key.astype(dt)
    value = value.astype(dt)
    log_decay = log_decay.astype(dt)
    beta = beta.astype(dt)
    query = query / np.maximum(
        np.linalg.norm(query, axis=-1, keepdims=True), 1e-12
    )
    key = key / np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1e-12)
    b, t, h, k = query.shape
    state = np.zeros((b, h, k, value.shape[-1]), dtype=dt)
    out = np.zeros((b, t, h, value.shape[-1]), dtype=dt)
    scale = 1.0 / np.sqrt(k)
    for token in range(t):
        state *= np.exp(log_decay[:, token])[..., None]
        pred = np.einsum("bhk,bhkv->bhv", key[:, token], state)
        delta = beta[:, token, :, None] * (value[:, token] - pred)
        state += np.einsum("bhk,bhv->bhkv", key[:, token], delta)
        out[:, token] = np.einsum(
            "bhk,bhkv->bhv", query[:, token] * scale, state
        )
    return out


def run(force_slow: bool, args, chunk_size):
    real = kda_mod._factored_decay_is_safe
    if force_slow:
        kda_mod._factored_decay_is_safe = lambda *a, **k: False
    try:
        return kda_recurrence(*args, chunk_size).detach().numpy()
    finally:
        kda_mod._factored_decay_is_safe = real


CHUNK = 8
print(f"chunk_size={CHUNK}, both paths vs a float64 sequential reference\n")
print(f"{'saturated':>10} {'factored err':>14} {'pairwise err':>14} "
      f"{'ratio':>8}")
print("-" * 50)

for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    rng = np.random.default_rng(5)
    shape = (2, 32, 4, 16)
    q = rng.normal(size=shape).astype(np.float32)
    k = rng.normal(size=shape).astype(np.float32)
    v = rng.normal(size=shape).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=shape[:-1]).astype(np.float32)
    decay = -rng.uniform(0.001, 2.0, size=shape).astype(np.float32)
    mask = rng.random(size=shape) < fraction
    decay[mask] = KDA_LOG_DECAY_FLOOR

    args = tuple(
        torch.from_numpy(a) for a in (q, k, v, decay, beta)
    )
    expected = reference(q, k, v, decay, beta)

    fast = run(False, args, CHUNK)
    slow = run(True, args, CHUNK)

    err_fast = float(np.max(np.abs(fast - expected)))
    err_slow = float(np.max(np.abs(slow - expected)))
    ratio = err_fast / err_slow if err_slow else float("inf")
    print(f"{fraction:>10.0%} {err_fast:>14.3e} {err_slow:>14.3e} "
          f"{ratio:>7.1f}x")

print("\nSanity: the two paths must agree with each other to the")
print("tolerance the parity suite asserts (rtol=1e-4).")
rng = np.random.default_rng(5)
shape = (2, 32, 4, 16)
q = rng.normal(size=shape).astype(np.float32)
k = rng.normal(size=shape).astype(np.float32)
v = rng.normal(size=shape).astype(np.float32)
beta = rng.uniform(0.0, 1.0, size=shape[:-1]).astype(np.float32)
decay = -rng.uniform(0.001, 2.0, size=shape).astype(np.float32)
decay[rng.random(size=shape) < 0.25] = KDA_LOG_DECAY_FLOOR
args = tuple(torch.from_numpy(a) for a in (q, k, v, decay, beta))
fast, slow = run(False, args, CHUNK), run(True, args, CHUNK)
print(f"  max |fast - slow| = {float(np.max(np.abs(fast - slow))):.3e}")
