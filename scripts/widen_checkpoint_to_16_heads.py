"""Widen checkpoint from 8 heads to 16 heads and strip policy head biases.

Usage:
    python scripts/widen_checkpoint_to_16_heads.py [--config docs/kda_split.textproto]
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import shutil
import sys
from typing import Any

import torch
from google.protobuf import text_format

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[1] / "src")
    if "__file__" in globals()
    else "src",
)

from lczero_training.directml import checkpoint as checkpoint_io
from lczero_training.directml.model import LczeroModel
from proto.root_config_pb2 import RootConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def migrate_weights(
    state_8: dict[str, Any],
    model_16: LczeroModel,
) -> dict[str, Any]:
    """Map 8-head state dict into 16-head model."""
    state_16 = model_16.state_dict()
    new_state: dict[str, Any] = {}

    for name, p_16 in state_16.items():
        if name not in state_8:
            logger.info("New parameter not in 8-head checkpoint: %s (shape %s)", name, p_16.shape)
            new_state[name] = p_16.clone()
            continue

        p_8 = state_8[name]
        if p_16.shape == p_8.shape:
            new_state[name] = p_8.clone()
        else:
            logger.info("Widening parameter: %s (%s -> %s)", name, list(p_8.shape), list(p_16.shape))
            expanded = p_16.clone()

            # Case 1: KDA Q, K, V, decay_b, gate_b (out_features: 256 -> 512)
            if p_16.ndim == 2 and p_16.shape[0] == 512 and p_8.shape[0] == 256 and p_16.shape[1] == p_8.shape[1]:
                for d in range(8):
                    old_slice = p_8[d * 32 : (d + 1) * 32, :]
                    expanded[2 * d * 32 : (2 * d + 1) * 32, :] = old_slice
            elif p_16.ndim == 1 and p_16.shape[0] == 512 and p_8.shape[0] == 256:
                for d in range(8):
                    old_slice = p_8[d * 32 : (d + 1) * 32]
                    expanded[2 * d * 32 : (2 * d + 1) * 32] = old_slice

            # Case 2: output_dense.weight (in_features: 256 -> 512, out_features: 128)
            elif p_16.ndim == 2 and p_16.shape[0] == 128 and p_8.shape[0] == 128 and p_16.shape[1] == 512 and p_8.shape[1] == 256:
                for d in range(8):
                    old_slice = p_8[:, d * 32 : (d + 1) * 32]
                    expanded[:, 2 * d * 32 : (2 * d + 1) * 32] = old_slice
                    expanded[:, (2 * d + 1) * 32 : (2 * d + 2) * 32] = 0.0

            # Case 3: beta.weight (shape (16, 128) vs (8, 128))
            elif p_16.ndim == 2 and p_16.shape[0] == 16 and p_8.shape[0] == 8 and p_16.shape[1] == p_8.shape[1]:
                for d in range(8):
                    expanded[2 * d, :] = p_8[d, :]
            elif p_16.ndim == 1 and p_16.shape[0] == 16 and p_8.shape[0] == 8:
                for d in range(8):
                    expanded[2 * d] = p_8[d]

            # Case 4: log_decay (a_log, dt_bias) - let freshly initialized 16-head log decay stay
            elif "log_decay" in name:
                pass

            # Case 5: Smolgen dense2 / ln2 (256 -> 512)
            elif "smolgen.dense2.weight" in name:
                expanded[:256, :] = p_8
            elif "smolgen.dense2.bias" in name:
                expanded[:256] = p_8
            elif "smolgen.ln2" in name:
                expanded[:256] = p_8
            else:
                logger.warning("Unhandled parameter resize for %s", name)

            new_state[name] = expanded

    return new_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="docs/kda_split.textproto",
        help="Path to training config file",
    )
    args = parser.parse_args()

    config = RootConfig()
    with open(args.config, "r", encoding="utf-8") as handle:
        text_format.Parse(handle.read(), config)

    assert config.model.encoder.heads == 16, (
        f"Expected encoder.heads == 16 in {args.config}, found {config.model.encoder.heads}"
    )

    ckpt_dir = pathlib.Path(config.training.checkpoint.path)
    files = checkpoint_io._checkpoint_files(ckpt_dir)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    latest_step, latest_path = files[-1]
    logger.info("Found latest checkpoint at step %d: %s", latest_step, latest_path)

    backup_path = latest_path.with_suffix(".pt.bak8head")
    if not backup_path.exists():
        shutil.copy2(latest_path, backup_path)
        logger.info("Created backup of original 8-head checkpoint at %s", backup_path)
    else:
        logger.info("Backup %s already exists", backup_path)

    # Load 8-head checkpoint raw payload
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    state_8 = payload["model_state"]

    # Build 16-head model
    model_16 = LczeroModel(config.model)
    migrated_state = migrate_weights(state_8, model_16)

    checkpoint_io.load_state_dict_into(model_16, migrated_state)
    logger.info("Successfully validated migrated state_dict into 16-head model!")

    # Verify forward pass
    dummy = torch.zeros(1, 112, 8, 8)
    out = model_16(dummy)
    logger.info("Verification forward pass successful! Policy output shape: %s", list(out.policy["vanilla"].shape))

    # Build new Checkpoint object with reset optimizer state
    new_digest = checkpoint_io.config_digest(config)
    new_checkpoint = checkpoint_io.Checkpoint(
        step=latest_step,
        model_state=migrated_state,
        optimizer_state=None,  # Reset moments to allow re-allocation matching new parameter sizes
        config_digest=new_digest,
        rng_state=payload.get("rng_state"),
        version=payload.get("version", checkpoint_io.CHECKPOINT_VERSION),
    )

    checkpoint_io.save(ckpt_dir, new_checkpoint, max_to_keep=config.training.checkpoint.max_to_keep)
    logger.info("Wrote migrated 16-head checkpoint to %s (config_digest: %s)", latest_path, new_digest[:16])


if __name__ == "__main__":
    main()
