import importlib.metadata
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np
import tensorflow as tf


TF_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TF_DIR))

# tfprocess only uses this optional dependency while reporting model FLOPs.
sys.modules.setdefault("tensorflow_models", types.SimpleNamespace())

from net import Net, pb
from tfprocess import KDA_LOG_DECAY_FLOOR, KDA_TRAVERSALS, TFProcess


def numpy_recurrent_kda(q, k, v, log_decay, beta):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
    batch_size, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = np.zeros(
        [batch_size, heads, key_dim, value_dim], dtype=np.float32)
    outputs = np.zeros(
        [batch_size, tokens, heads, value_dim], dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(key_dim))

    for token in range(tokens):
        state *= np.exp(log_decay[:, token])[..., None]
        prediction = np.einsum("bhk,bhkv->bhv", k[:, token], state)
        delta = beta[:, token, :, None] * (v[:, token] - prediction)
        state += np.einsum("bhk,bhv->bhkv", k[:, token], delta)
        outputs[:, token] = np.einsum(
            "bhk,bhkv->bhv", q[:, token] * scale, state)
    return outputs


def make_process():
    process = TFProcess.__new__(TFProcess)
    process.omit_qkv_biases = True
    process.omit_other_biases = False
    process.kda_key_dim = 4
    process.kda_value_dim = 4
    process.kda_gate_rank = 4
    process.kda_directions = list(KDA_TRAVERSALS)
    process.kda_output_gate = True
    process.kda_local_conv = False
    process.model_dtype = tf.float32
    process.use_smolgen = False
    process.use_logit_gating = False
    return process


class KDATest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.mixed_precision.set_global_policy("float32")
        tf.random.set_seed(7)

    def tearDown(self):
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_recurrence_matches_numpy(self):
        generator = np.random.default_rng(7)
        q = generator.normal(size=[2, 5, 2, 3]).astype(np.float32)
        k = generator.normal(size=[2, 5, 2, 3]).astype(np.float32)
        v = generator.normal(size=[2, 5, 2, 4]).astype(np.float32)
        log_decay = -generator.uniform(
            0.001, 2.0, size=[2, 5, 2, 3]).astype(np.float32)
        beta = generator.uniform(0.0, 1.0, size=[2, 5, 2]).astype(np.float32)

        actual = make_process().recurrent_kda(
            q, k, v, log_decay, beta).numpy()
        expected = numpy_recurrent_kda(q, k, v, log_decay, beta)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_recurrence_carries_state_across_chunks(self):
        # A full board exercises several chunks of the parallel form, so the
        # inter-chunk state hand-off is covered as well as the intra-chunk solve.
        generator = np.random.default_rng(11)
        q = generator.normal(size=[3, 64, 4, 8]).astype(np.float32)
        k = generator.normal(size=[3, 64, 4, 8]).astype(np.float32)
        v = generator.normal(size=[3, 64, 4, 8]).astype(np.float32)
        beta = generator.uniform(0.0, 1.0, size=[3, 64, 4]).astype(np.float32)

        for decay_scale in (2.0, 20.0):
            with self.subTest(decay_scale=decay_scale):
                log_decay = -generator.uniform(
                    0.001, decay_scale, size=[3, 64, 4, 8]).astype(np.float32)
                actual = make_process().recurrent_kda(
                    q, k, v, log_decay, beta).numpy()
                expected = numpy_recurrent_kda(
                    q, k, v,
                    np.maximum(log_decay, KDA_LOG_DECAY_FLOOR), beta)
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-4, atol=1e-5)

    def test_traversals_are_invertible_permutations(self):
        squares = np.arange(64)
        for direction, order in KDA_TRAVERSALS.items():
            with self.subTest(direction=direction):
                self.assertEqual(sorted(order), list(range(64)))
                restored = squares[np.asarray(order)][np.argsort(order)]
                np.testing.assert_array_equal(restored, squares)

    def test_extreme_decay_and_zero_inputs_are_finite(self):
        shape = [2, 7, 2, 3]
        q = tf.zeros(shape, dtype=tf.float16)
        k = tf.zeros(shape, dtype=tf.float16)
        v = tf.zeros([2, 7, 2, 4], dtype=tf.float16)
        beta = tf.zeros([2, 7, 2], dtype=tf.float16)
        for decay in (-1e-7, -1e6):
            with self.subTest(decay=decay):
                log_decay = tf.fill(shape, tf.cast(decay, tf.float16))
                output = make_process().recurrent_kda(
                    q, k, v, log_decay, beta)
                self.assertEqual(output.dtype, tf.float16)
                self.assertTrue(np.all(np.isfinite(output.numpy())))

    def test_saturating_decay_keeps_outputs_and_gradients_finite(self):
        # The parallel form splits exp(cumulative_i - cumulative_j) into two
        # factors that individually reach the edge of the float32 range once the
        # decay floor binds, so check nothing overflows into the gradients.
        generator = np.random.default_rng(3)
        shape = [2, 64, 4, 16]
        q = tf.constant(generator.normal(size=shape).astype(np.float32))
        k = tf.constant(generator.normal(size=shape).astype(np.float32))
        v = tf.constant(generator.normal(size=shape).astype(np.float32))
        beta = tf.constant(
            generator.uniform(0.1, 1.0, size=shape[:3]).astype(np.float32))
        log_decay = tf.Variable(tf.fill(shape, -1e4))

        process = make_process()
        with tf.GradientTape() as tape:
            output = process.recurrent_kda(q, k, v, log_decay, beta)
            loss = tf.reduce_sum(tf.square(output))
        gradient = tape.gradient(loss, log_decay)

        self.assertTrue(np.all(np.isfinite(output.numpy())))
        self.assertTrue(np.all(np.isfinite(gradient.numpy())))

    def test_kda_output_shape_names_and_gradients(self):
        expected_names = [
            "/kda/wq/", "/kda/wk/", "/kda/wv/",
            "/kda/decay_a/", "/kda/decay_b/", "/kda/beta/",
            "/kda/a_log", "/kda/dt_bias", "/kda/gate_a/",
            "/kda/gate_b/", "/kda/dense/",
        ]
        policies = ["float32"]
        try:
            importlib.metadata.version("tensorflow-directml-plugin")
        except importlib.metadata.PackageNotFoundError:
            policies.append("mixed_float16")

        for policy in policies:
            with self.subTest(policy=policy):
                tf.keras.backend.clear_session()
                tf.keras.mixed_precision.set_global_policy(policy)
                process = make_process()
                inputs = tf.keras.Input(shape=[64, 16])
                outputs = process.kda(
                    inputs, emb_size=16, num_heads=4,
                    initializer="glorot_normal", name="encoder_1/kda")
                model = tf.keras.Model(inputs=inputs, outputs=outputs)

                with tf.GradientTape() as tape:
                    output = model(tf.random.normal([2, 64, 16]))
                    loss = tf.reduce_sum(tf.square(tf.cast(output, tf.float32)))
                gradients = tape.gradient(loss, model.trainable_variables)

                self.assertEqual(output.shape, [2, 64, 16])
                self.assertTrue(np.all(np.isfinite(output.numpy())))
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(np.all(np.isfinite(gradient.numpy()))
                                    for gradient in gradients))
                variable_names = [variable.name
                                  for variable in model.trainable_variables]
                for expected_name in expected_names:
                    self.assertTrue(
                        any(expected_name in name for name in variable_names),
                        "Missing variable name containing {}".format(
                            expected_name))

    def test_kda_protobuf_round_trip(self):
        process = make_process()
        inputs = tf.keras.Input(shape=[64, 16])
        outputs = process.kda(
            inputs, emb_size=16, num_heads=4,
            initializer="glorot_normal", name="encoder_1/kda")
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        net = Net()
        net.set_networkformat(
            pb.NetworkFormat.NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT)
        net.set_headcount(4)
        net.set_kda_directions(list(KDA_TRAVERSALS))
        net.set_encoder_mixer(
            0, "kda", key_dim=4, value_dim=4, gate_rank=4,
            output_gate=True)
        original = {weight.name: weight.numpy()
                    for weight in model.weights}
        net.fill_net_v2(list(original.items()))

        with tempfile.TemporaryDirectory() as directory:
            filename = str(pathlib.Path(directory) / "kda.pb.gz")
            net.save_proto(filename)
            restored_net = Net()
            restored_net.parse_proto(filename)

        network_format = restored_net.pb.format.network_format
        encoder = restored_net.pb.weights.encoder[0]
        self.assertEqual(
            network_format.network,
            pb.NetworkFormat.NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT)
        self.assertEqual(
            encoder.mixer, pb.Weights.EncoderLayer.MIXER_KDA)
        self.assertEqual(
            list(network_format.kda_directions), [1, 2, 3, 4])
        self.assertEqual(
            (encoder.kda.key_dim, encoder.kda.value_dim,
             encoder.kda.gate_rank), (4, 4, 4))
        self.assertFalse(encoder.kda.output_rms_norm)
        self.assertGreaterEqual(restored_net.pb.min_version.minor, 33)

        restored = restored_net.get_weights_v2(list(original))
        for name, expected in original.items():
            if expected.ndim == 2 and not (
                    "/kda/a_log:" in name or "/kda/dt_bias:" in name):
                expected = expected.T
            expected = expected.reshape(-1)
            quantization_step = max(
                float(np.ptp(expected)) / 0xffff, 1e-6)
            np.testing.assert_allclose(
                restored[name], expected, rtol=0,
                atol=quantization_step,
                err_msg=name)

    def test_kda_local_conv_shape_and_gradients(self):
        process = make_process()
        process.kda_local_conv = True
        inputs = tf.keras.Input(shape=[64, 16])
        outputs = process.kda(
            inputs, emb_size=16, num_heads=4,
            initializer="glorot_normal", name="encoder_1/kda")
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        with tf.GradientTape() as tape:
            output = model(tf.random.normal([2, 64, 16]))
            loss = tf.reduce_sum(tf.square(output))
        gradients = tape.gradient(loss, model.trainable_variables)

        self.assertEqual(output.shape, [2, 64, 16])
        self.assertTrue(np.all(np.isfinite(output.numpy())))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(np.all(np.isfinite(gradient.numpy()))
                            for gradient in gradients))
        variable_names = [variable.name
                          for variable in model.trainable_variables]
        self.assertTrue(any("/kda/local_conv/depthwise_kernel" in name
                            for name in variable_names))

    def test_kda_local_conv_only_sees_3x3_neighborhood(self):
        process = make_process()
        process.kda_local_conv = True
        inputs = tf.keras.Input(shape=[64, 16])
        outputs = process.kda(
            inputs, emb_size=16, num_heads=4,
            initializer="glorot_normal", name="encoder_1/kda")
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        base = tf.random.normal([1, 64, 16])
        perturbed = base.numpy().copy()
        # Square (rank=3, file=3) -> token 3*8+3 = 27, outside the 3x3
        # neighborhood of token 0 (square (rank=0, file=0)).
        perturbed[0, 27, :] += 5.0
        output_base = model(base).numpy()
        output_perturbed = model(tf.constant(perturbed)).numpy()

        np.testing.assert_allclose(
            output_base[0, 0], output_perturbed[0, 0], rtol=1e-5, atol=1e-5)

    def test_kda_local_conv_protobuf_round_trip(self):
        process = make_process()
        process.kda_local_conv = True
        inputs = tf.keras.Input(shape=[64, 16])
        outputs = process.kda(
            inputs, emb_size=16, num_heads=4,
            initializer="glorot_normal", name="encoder_1/kda")
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        net = Net()
        net.set_networkformat(
            pb.NetworkFormat.NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT)
        net.set_headcount(4)
        net.set_kda_directions(list(KDA_TRAVERSALS))
        net.set_encoder_mixer(
            0, "kda", key_dim=4, value_dim=4, gate_rank=4,
            output_gate=True, local_conv=True)
        original = {weight.name: weight.numpy()
                    for weight in model.weights}
        net.fill_net_v2(list(original.items()))

        with tempfile.TemporaryDirectory() as directory:
            filename = str(pathlib.Path(directory) / "kda.pb.gz")
            net.save_proto(filename)
            restored_net = Net()
            restored_net.parse_proto(filename)

        encoder = restored_net.pb.weights.encoder[0]
        self.assertTrue(encoder.kda.local_conv)
        self.assertGreaterEqual(restored_net.pb.min_version.minor, 34)

        restored = restored_net.get_weights_v2(list(original))
        for name, expected in original.items():
            if expected.ndim == 4:
                # Conv weights: TF [kh, kw, in, out] -> Leela [out, in, kh, kw].
                expected = np.transpose(expected, axes=[3, 2, 0, 1])
            elif expected.ndim == 2 and not (
                    "/kda/a_log:" in name or "/kda/dt_bias:" in name):
                expected = expected.T
            expected = expected.reshape(-1)
            quantization_step = max(
                float(np.ptp(expected)) / 0xffff, 1e-6)
            np.testing.assert_allclose(
                restored[name], expected, rtol=0,
                atol=quantization_step,
                err_msg=name)

    def test_legacy_kda_defaults_to_output_rms_norm(self):
        net = Net()
        encoder = net.pb.weights.encoder.add()
        encoder.mixer = pb.Weights.EncoderLayer.MIXER_KDA

        self.assertTrue(encoder.kda.output_rms_norm)
        self.assertFalse(encoder.kda.HasField("output_rms_norm"))

    def test_mha_names_and_shapes_remain_unchanged(self):
        process = make_process()
        inputs = tf.keras.Input(shape=[64, 16])
        outputs, _ = process.mha(
            inputs, emb_size=16, d_model=16, num_heads=4,
            initializer="glorot_normal", name="encoder_1/mha")
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        variables = {variable.name: tuple(variable.shape)
                     for variable in model.trainable_variables}

        self.assertEqual(model.output_shape, (None, None, 16))
        self.assertFalse(any("/kda/" in name for name in variables))
        for layer in ("wq", "wk", "wv", "dense"):
            kernel = next(shape for name, shape in variables.items()
                          if "/mha/{}/kernel".format(layer) in name)
            self.assertEqual(kernel, (16, 16))


if __name__ == "__main__":
    unittest.main()