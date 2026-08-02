"""Times a forward+backward pass of the configured model at several micro-batch
sizes, to pick `num_batch_splits`. Run: python tests/bench_microbatch.py CONFIG
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


def main(config_path, sizes):
    cfg = yaml.safe_load(open(config_path))
    total = cfg["training"]["batch_size"]
    cfg["training"]["num_batch_splits"] = 1
    process = TFProcess(cfg)
    process.init_net()
    model = process.model
    variables = model.trainable_weights

    @tf.function
    def step(x):
        with tf.GradientTape() as tape:
            outputs = model(x, training=True)
            loss = tf.add_n(
                [tf.reduce_mean(tf.square(tf.cast(o, tf.float32)))
                 for o in tf.nest.flatten(outputs)])
        return tape.gradient(loss, variables)

    print("total batch {}".format(total))
    for size in sizes:
        x = tf.random.uniform([size, 112, 8, 8], dtype=tf.float32)
        for _ in range(3):
            step(x)
        _ = [g.numpy() for g in step(x) if g is not None][0]
        repeats = max(2, min(10, 512 // size))
        start = time.perf_counter()
        for _ in range(repeats):
            grads = step(x)
        _ = [g.numpy() for g in grads if g is not None][0]
        elapsed = (time.perf_counter() - start) / repeats
        print("micro-batch {:5d} (splits {:3d}): {:7.1f} ms/micro-batch, "
              "{:8.0f} positions/s, {:7.0f} ms per {} positions".format(
                  size, total // size, elapsed * 1e3, size / elapsed,
                  elapsed * 1e3 * (total / size), total))


if __name__ == "__main__":
    config = sys.argv[1]
    candidates = [int(a) for a in sys.argv[2:]] or [16, 32, 64, 128, 256]
    main(config, candidates)
