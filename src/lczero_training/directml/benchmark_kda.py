"""Phase 4 KDA performance gate.

Benchmarks three complete KDA mixers -- the KDA half of the target
4-block hybrid tower in docs/example_kda_real_import.textproto -- through a
full forward and backward pass, on DirectML and on JAX CPU, and reports
peak shared GPU memory.

This is a hard continuation gate for the port (see
docs/directml_training_port.md, Phase 4). DirectML does not support
torch.compile in this configuration and the chunkwise KDA implementation
launches many eager operations, so if the complete KDA body is not
materially faster than CPU training there is nothing to gain from Phases 5
onward.

    uv run lc0-directml-bench-kda
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn

from proto import model_config_pb2

# Must precede any backward pass in the process: the autograd engine sizes
# its per-device ready queues on first use, and PrivateUse1 is only counted
# if torch_directml has already been imported. See device.ensure_initialized.
from . import device as dml_device
from . import layers
from .kda import KdaMixer

dml_device.ensure_initialized()

# The KDA settings of docs/example_kda_real_import.textproto.
TARGET_EMBEDDING_SIZE = 128
TARGET_HEADS = 8
TARGET_BLOCKS = 4  # tower depth, for the DeepNorm constants
KDA_BLOCKS = 3  # the three MIXER_KDA entries of the mixer_pattern
DEFAULT_BATCHES = (4, 8, 16, 32)


def target_config(chunk_size: int) -> model_config_pb2.KdaConfig:
    return model_config_pb2.KdaConfig(
        key_dim=32,
        value_dim=32,
        gate_rank=32,
        directions=[
            "rank_forward",
            "rank_reverse",
            "file_forward",
            "file_reverse",
            "diag_forward",
            "diag_reverse",
            "anti_diag_forward",
            "anti_diag_reverse",
        ],
        output_gate=True,
        output_rms_norm=False,
        local_conv=True,
        chunk_size=chunk_size,
    )


class TorchKdaStack(nn.Module):
    """``KDA_BLOCKS`` mixers chained with the encoder's DeepNorm residual."""

    def __init__(self, config: model_config_pb2.KdaConfig, blocks: int):
        super().__init__()
        beta = layers.deepnorm_beta(TARGET_BLOCKS)
        self.alpha = layers.deepnorm_alpha(TARGET_BLOCKS)
        self.mixers = nn.ModuleList(
            [
                KdaMixer(
                    in_features=TARGET_EMBEDDING_SIZE,
                    config=config,
                    heads=TARGET_HEADS,
                    deepnorm_beta=beta,
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mixer in self.mixers:
            x = layers.deepnorm_residual(x, mixer(x), self.alpha)
        return x


def _percentiles(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    return median, ordered[0]


def benchmark_torch(
    device: torch.device,
    config: model_config_pb2.KdaConfig,
    batch: int,
    blocks: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float | None]:
    """Median and best milliseconds per forward+backward, and memory used."""
    stack = TorchKdaStack(config, blocks).to(device)
    x = torch.randn(
        batch, 64, TARGET_EMBEDDING_SIZE, device=device, requires_grad=True
    )

    def step(probe: list[float] | None = None) -> None:
        stack.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        loss = stack(x).square().sum()
        if probe is not None:
            # Sample with the forward graph still alive -- every saved
            # activation is resident here, which is the actual peak.
            float(loss.detach().cpu())
            _record(probe, device)
        loss.backward()
        # DirectML queues work asynchronously; pulling a scalar to the host
        # is what actually waits for the backward pass to retire.
        float(loss.detach().cpu())
        if probe is not None:
            _record(probe, device)

    for _ in range(warmup):
        step()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        step()
        samples.append((time.perf_counter() - start) * 1000.0)

    # Memory is measured in its own untimed pass: the probe forces two extra
    # host syncs, which would distort the timings above.
    probe: list[float] = []
    step(probe)
    peak_memory = max(probe) if probe else None

    median, best = _percentiles(samples)
    return median, best, peak_memory


def _record(probe: list[float], device: torch.device) -> None:
    current = _used_gpu_memory(device)
    if current is not None:
        probe.append(current)


def _used_gpu_memory(device: torch.device) -> float | None:
    """Shared GPU memory currently in use, in MiB.

    ``torch_directml.gpu_memory`` returns one fill fraction per tile of
    ``mb_per_tile`` megabytes, not a single total, so the sum over the list
    is the number of megabytes actually resident. Iris Xe is an integrated
    adapter, so this is shared system memory.
    """
    if device.type != "privateuseone":
        return None
    try:
        import torch_directml

        return float(sum(torch_directml.gpu_memory(device.index or 0, 1)))
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def benchmark_jax_cpu(
    config: model_config_pb2.KdaConfig,
    batch: int,
    blocks: int,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Same workload under jitted JAX on the CPU, vmapped over the batch."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from lczero_training.model.kda import KdaMixer as JaxKdaMixer

    beta = layers.deepnorm_beta(TARGET_BLOCKS)
    alpha = layers.deepnorm_alpha(TARGET_BLOCKS)

    class JaxKdaStack(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            # nnx.List, not a bare list: NNX (>=0.12) refuses to treat a
            # plain list of Modules as pytree data.
            self.mixers = nnx.List(
                [
                    JaxKdaMixer(
                        in_features=TARGET_EMBEDDING_SIZE,
                        config=config,
                        heads=TARGET_HEADS,
                        deepnorm_beta=beta,
                        rngs=rngs,
                    )
                    for _ in range(blocks)
                ]
            )

        def __call__(self, x):
            for mixer in self.mixers:
                x = x + mixer(x) * alpha
            return x

    stack = JaxKdaStack(nnx.Rngs(0))
    graphdef, state = nnx.split(stack)

    @jax.jit
    def step(state, batch_x):
        def loss_fn(state):
            model = nnx.merge(graphdef, state)
            out = jax.vmap(model)(batch_x)
            return jnp.sum(jnp.square(out))

        return jax.grad(loss_fn)(state)

    rng = np.random.default_rng(0)
    x = jnp.asarray(
        rng.normal(size=(batch, 64, TARGET_EMBEDDING_SIZE)).astype(np.float32)
    )

    for _ in range(warmup):
        jax.block_until_ready(step(state, x))

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(step(state, x))
        samples.append((time.perf_counter() - start) * 1000.0)

    return _percentiles(samples)


def _format_memory(used: float | None) -> str:
    return "n/a" if used is None else f"{used:.0f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
        help="Batch sizes to benchmark.",
    )
    parser.add_argument("--blocks", type=int, default=KDA_BLOCKS)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--device", default=None, help="Defaults to directml:0."
    )
    parser.add_argument(
        "--skip-jax", action="store_true", help="Skip the JAX CPU baseline."
    )
    args = parser.parse_args()

    config = target_config(args.chunk_size)
    device = dml_device.resolve_device(args.device)
    print(f"Device: {device} ({dml_device.adapter_name(0)})")
    print(
        f"Workload: {args.blocks} KDA mixers, embedding "
        f"{TARGET_EMBEDDING_SIZE}, {TARGET_HEADS} heads, "
        f"chunk_size {args.chunk_size}, forward + backward"
    )
    print(
        f"{args.warmup} warmup + {args.iterations} timed iterations "
        f"per batch size\n"
    )

    header = (
        f"{'batch':>6}  {'DirectML ms':>12}  {'JAX CPU ms':>11}  "
        f"{'speedup':>8}  {'GPU MiB':>8}"
    )
    print(header)
    print("-" * len(header))

    speedups = []
    for batch in args.batches:
        torch_median, _torch_best, memory = benchmark_torch(
            device, config, batch, args.blocks, args.warmup, args.iterations
        )
        if args.skip_jax:
            print(
                f"{batch:>6}  {torch_median:>12.1f}  {'-':>11}  "
                f"{'-':>8}  {_format_memory(memory):>8}"
            )
            continue
        jax_median, _jax_best = benchmark_jax_cpu(
            config, batch, args.blocks, args.warmup, args.iterations
        )
        speedup = jax_median / torch_median
        speedups.append(speedup)
        print(
            f"{batch:>6}  {torch_median:>12.1f}  {jax_median:>11.1f}  "
            f"{speedup:>7.2f}x  {_format_memory(memory):>8}"
        )

    if speedups:
        best = max(speedups)
        print(
            f"\nBest DirectML speedup over JAX CPU: {best:.2f}x "
            f"(at batch {args.batches[speedups.index(best)]})"
        )
        print(
            "Phase 4 gate: DirectML must be materially faster than CPU "
            "training. " + ("PASS" if best > 1.0 else "FAIL")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
