"""Migrate LC0 DirectML checkpoint by stripping KDA local depthwise convolutions.

Usage:
    python scripts/migrate_disable_local_conv.py [--config tf/configs/kda-t1.textproto] [--checkpoint <path>]
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
logger = logging.getLogger("migrate_local_conv")


def migrate_checkpoint_remove_local_conv(
    config_path: str,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
) -> pathlib.Path:
    config = RootConfig()
    with open(config_path, "r", encoding="utf-8") as handle:
        text_format.Parse(handle.read(), config)

    if config.model.encoder.kda.local_conv:
        logger.warning(
            "config.model.encoder.kda.local_conv is currently True in %s; forcing to False for migration.",
            config_path,
        )
        config.model.encoder.kda.local_conv = False

    ckpt_dir = pathlib.Path(output_dir or config.training.checkpoint.path)

    if checkpoint_path:
        target_path = pathlib.Path(checkpoint_path)
        match = checkpoint_io._FILENAME_RE.match(target_path.name)
        step = int(match.group(1)) if match else 0
    else:
        source_dir = pathlib.Path(config.training.checkpoint.path)
        files = checkpoint_io._checkpoint_files(source_dir)
        if not files:
            raise FileNotFoundError(f"No checkpoint files found in {source_dir}")
        step, target_path = files[-1]

    logger.info("Selected checkpoint for migration at step %d: %s", step, target_path)

    # 1. Create a safe backup
    backup_path = target_path.with_suffix(".pt.bak_local_conv")
    if not backup_path.exists():
        shutil.copy2(target_path, backup_path)
        logger.info("Created backup of original checkpoint at %s", backup_path)
    else:
        logger.info("Backup already exists at %s", backup_path)

    # 2. Load checkpoint payload
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    old_state = payload["model_state"]
    old_digest = payload.get("config_digest", "none")

    # 3. Filter out local_conv weights
    stripped_keys = []
    filtered_state: dict[str, Any] = {}
    for k, v in old_state.items():
        if "local_conv" in k:
            stripped_keys.append(k)
        else:
            filtered_state[k] = v

    logger.info("Stripped %d local_conv parameters from state_dict:", len(stripped_keys))
    for k in stripped_keys:
        logger.info("  - Removed: %s (%s)", k, list(old_state[k].shape))

    # 4. Instantiate target model with local_conv = False
    new_model = LczeroModel(config.model)
    checkpoint_io.load_state_dict_into(new_model, filtered_state)
    logger.info("Successfully loaded filtered state_dict into LczeroModel (local_conv=False)!")

    # 5. Verification forward pass
    new_model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, 112, 8, 8)
        output = new_model(dummy_input)
        for head_name, tensor in output.policy.items():
            assert torch.isfinite(tensor).all(), f"Non-finite policy output in head {head_name}"
            logger.info("  Policy head '%s': output shape %s (finite=True)", head_name, list(tensor.shape))
        for head_name, val in output.value.items():
            if isinstance(val, tuple):
                for i, elem in enumerate(val):
                    if elem is not None:
                        assert torch.isfinite(elem).all(), f"Non-finite value output in head {head_name}[{i}]"
                        logger.info("  Value head '%s'[%d]: output shape %s (finite=True)", head_name, i, list(elem.shape))
            elif val is not None:
                assert torch.isfinite(val).all(), f"Non-finite value output in head {head_name}"
                logger.info("  Value head '%s': output shape %s (finite=True)", head_name, list(val.shape))
        for head_name, tensor in output.movesleft.items():
            assert torch.isfinite(tensor).all(), f"Non-finite movesleft output in head {head_name}"
            logger.info("  Movesleft head '%s': output shape %s (finite=True)", head_name, list(tensor.shape))

    logger.info("Forward-pass sanity check passed successfully on all heads!")

    # 6. Compute new config digest and package checkpoint
    new_digest = checkpoint_io.config_digest(config)
    logger.info("Config digest transition: %s -> %s", old_digest[:16], new_digest[:16])

    new_checkpoint = checkpoint_io.Checkpoint(
        step=step,
        model_state=filtered_state,
        optimizer_state=None,  # Reset moments to allow re-allocation matching new parameter sizes
        config_digest=new_digest,
        rng_state=payload.get("rng_state"),
        version=payload.get("version", checkpoint_io.CHECKPOINT_VERSION),
    )

    saved_path = checkpoint_io.save(ckpt_dir, new_checkpoint, max_to_keep=config.training.checkpoint.max_to_keep)
    logger.info("Migrated checkpoint successfully written to: %s", saved_path)
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="tf/configs/kda-t1.textproto",
        help="Path to training config textproto file",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to specific checkpoint file (.pt) to migrate",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write the migrated checkpoint to",
    )
    args = parser.parse_args()

    migrate_checkpoint_remove_local_conv(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
