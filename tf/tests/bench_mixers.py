"""Breaks a training step down by mixer type, to see where the time goes.
Run: python tests/bench_mixers.py CONFIG [MICRO_BATCH]
"""
import pathlib
import sys
import time
import types

import yaml

TF_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TF_DIR))
sys.modules.setdefault("tensorflow_models", types.SimpleNamespace())

import tensorflow as tf  # noqa: E402

from tfprocess import TFProcess  # noqa: E402


def time_step(cfg, size, repeats=6):
    tf.keras.backend.clear_session()
    process = TFProcess(cfg)
    process.init_net()
    variables = process.model.trainable_weights

    @tf.function
    def step(x):
        with tf.GradientTape() as tape:
            outputs = process.model(x, training=True)
            loss = tf.add_n(
                [tf.reduce_mean(tf.square(tf.cast(o, tf.float32)))
                 for o in tf.nest.flatten(outputs)])
        return tape.gradient(loss, variables)

    x = tf.random.uniform([size, 112, 8, 8], dtype=tf.float32)
    for _ in range(2):
        grads = step(x)
    _ = [g.numpy() for g in grads if g is not None][0]
    start = time.perf_counter()
    for _ in range(repeats):
        grads = step(x)
    _ = [g.numpy() for g in grads if g is not None][0]
    return (time.perf_counter() - start) / repeats


def main(config_path, size):
    base = yaml.safe_load(open(config_path))
    base["training"]["num_batch_splits"] = 1
    layers = base["model"]["encoder_layers"]

    variants = {
        "as configured": base["model"]["encoder_mixer_pattern"],
        "all mha": ["mha"],
        "all kda": ["kda"],
    }
    results = {}
    for label, pattern in variants.items():
        cfg = yaml.safe_load(open(config_path))
        cfg["training"]["num_batch_splits"] = 1
        cfg["model"]["encoder_mixer_pattern"] = pattern
        results[label] = time_step(cfg, size)
        print("RESULT {:>14}: {:8.1f} ms".format(label, results[label] * 1e3))

    cfg = yaml.safe_load(open(config_path))
    cfg["training"]["num_batch_splits"] = 1
    print("RESULT kda costs {:.1f} ms more than mha per layer, over {} "
          "layers".format(
              (results["all kda"] - results["all mha"]) * 1e3 / layers,
              layers))
    print("RESULT configured step is {:.1f} ms, {:.0f}% above an all-mha "
          "body".format(
              results["as configured"] * 1e3,
              100 * (results["as configured"] / results["all mha"] - 1)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16)
