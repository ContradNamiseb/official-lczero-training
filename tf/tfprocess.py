#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import os
import tensorflow as tf
import time
import bisect
import lc0_az_policy_map
import attention_policy_map as apm
import proto.net_pb2 as pb
from functools import reduce
import operator
import functools
from net import Net

try:
    import tensorflow_models as tfm
except ImportError:
    tfm = None

from keras import backend as K


def accelerator_device_type():
    """Return the device type this build exposes accelerators under.

    Intel Extension for TensorFlow registers Intel GPUs as PluggableDevices of
    type ``XPU``; CUDA, ROCm and DirectML builds all use ``GPU``.
    """
    if tf.config.list_physical_devices('XPU'):
        return 'XPU'
    return 'GPU'


def enable_memory_growth(device):
    # ITEX manages device memory itself and rejects this request.
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except (ValueError, tf.errors.UnimplementedError) as error:
        print(f"Memory growth unavailable for {device.name}: {error}")


def get_activation(activation):
    if isinstance(activation, str) or activation is None:
        try:
            return tf.keras.activations.get(activation)
        except:
            if activation == "mish":
                import tensorflow_addons as tfa
                return tfa.activations.mish
            else:
                raise ValueError(f"{activation=} not recognized")
    else:
        return activation


class Gating(tf.keras.layers.Layer):

    def __init__(self, name=None, additive=True, init_value=None, **kwargs):
        self.additive = additive
        if init_value is None:
            init_value = 0 if self.additive else 1
        self.init_value = init_value
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.gate = self.add_weight(name='gate',
                                    shape=input_shape[1:],
                                    constraint=tf.keras.constraints.NonNeg()
                                    if not self.additive else None,
                                    initializer=tf.constant_initializer(
                                        self.init_value),
                                    trainable=True)

    def call(self, inputs):
        return tf.add(inputs, self.gate) if self.additive else tf.multiply(
            inputs, self.gate)


def ma_gating(inputs, name):
    out = Gating(name=name + '/mult_gate', additive=False)(inputs)
    out = Gating(name=name + '/add_gate', additive=True)(out)
    return out


def square_relu(x):
    return tf.nn.relu(x) ** 2


KDA_TRAVERSALS = {
    "rank_forward": tuple(range(64)),
    "rank_reverse": tuple(reversed(range(64))),
    "file_forward": tuple(
        rank * 8 + file for file in range(8) for rank in range(8)),
    "file_reverse": tuple(
        reversed(tuple(
            rank * 8 + file for file in range(8) for rank in range(8)))),
    # Main diagonals (a1-h8 style, rank - file constant), grouped from the
    # a8 corner to the h1 corner and walked bottom-to-top within each one.
    "diag_forward": tuple(
        rank * 8 + file
        for diag in range(-7, 8)
        for rank in range(8)
        for file in [rank - diag] if 0 <= file < 8),
    "diag_reverse": tuple(
        reversed(tuple(
            rank * 8 + file
            for diag in range(-7, 8)
            for rank in range(8)
            for file in [rank - diag] if 0 <= file < 8))),
    # Anti-diagonals (a8-h1 style, rank + file constant).
    "anti_diag_forward": tuple(
        rank * 8 + file
        for diag in range(15)
        for rank in range(8)
        for file in [diag - rank] if 0 <= file < 8),
    "anti_diag_reverse": tuple(
        reversed(tuple(
            rank * 8 + file
            for diag in range(15)
            for rank in range(8)
            for file in [diag - rank] if 0 <= file < 8))),
}

# Tokens per chunk of the parallel KDA recurrence; 64 squares become 4 chunks.
# Measured fastest of 4/8/16/32/64 on DirectML: bigger chunks trade extra dense
# work for less sequential depth, and 16 sits at the turning point.
KDA_CHUNK_SIZE = 16
# Per-token log decay floor. The parallel form needs exp(+/- cumulative/2) to
# stay in the float32 exponent range, which caps this at 176 / KDA_CHUNK_SIZE.
# Retaining exp(-10) of the state per token is already indistinguishable from
# total forgetting, so lc0's unclamped sequential recurrence still matches.
KDA_LOG_DECAY_FLOOR = -10.0


class KDALogDecay(tf.keras.layers.Layer):
    def __init__(self, heads, key_dim, **kwargs):
        super(KDALogDecay, self).__init__(**kwargs)
        self.heads = heads
        self.key_dim = key_dim

    def build(self, input_shape):
        a_log = np.log(np.linspace(1.0, 16.0, self.heads,
                                   dtype=np.float32))[:, None]
        initial_dt = np.exp(np.linspace(
            np.log(0.001), np.log(0.1), self.key_dim,
            dtype=np.float32))
        dt_bias = np.log(np.expm1(initial_dt))[None, :]
        dt_bias = np.repeat(dt_bias, self.heads, axis=0)
        self.a_log = self.add_weight(
            name="a_log", shape=[self.heads, 1],
            initializer=tf.constant_initializer(a_log), trainable=True)
        self.dt_bias = self.add_weight(
            name="dt_bias", shape=[self.heads, self.key_dim],
            initializer=tf.constant_initializer(dt_bias), trainable=True)

    def call(self, raw_decay):
        raw_decay = tf.cast(raw_decay, tf.float32)
        a_log = tf.cast(self.a_log, tf.float32)
        dt_bias = tf.cast(self.dt_bias, tf.float32)
        return (-tf.exp(a_log)[None, None, :, :] *
                tf.nn.softplus(
                    raw_decay + dt_bias[None, None, :, :]))


class KDARecurrence(tf.keras.layers.Layer):
    """Chunkwise-parallel form of the gated delta rule.

    Equivalent to scanning the recurrence one token at a time, but each chunk is
    solved in closed form with dense matmuls, so the sequential depth drops from
    `tokens` to `tokens / chunk_size`.
    """

    def __init__(self, chunk_size=KDA_CHUNK_SIZE, **kwargs):
        super(KDARecurrence, self).__init__(**kwargs)
        self.chunk_size = chunk_size

    def call(self, inputs):
        q, k, v, log_decay, beta = inputs
        q = tf.math.l2_normalize(tf.cast(q, tf.float32), axis=-1)
        k = tf.math.l2_normalize(tf.cast(k, tf.float32), axis=-1)
        v = tf.cast(v, tf.float32)
        log_decay = tf.maximum(
            tf.cast(log_decay, tf.float32), KDA_LOG_DECAY_FLOOR)
        beta = tf.cast(beta, tf.float32)[..., None]

        tokens, heads, key_dim = q.shape[1], q.shape[2], q.shape[3]
        value_dim = v.shape[3]
        chunk = self.chunk_size
        padding = -tokens % chunk
        if padding:
            # Zero keys and betas make padded tokens leave the state untouched.
            pad4 = [[0, 0], [0, padding], [0, 0], [0, 0]]
            q = tf.pad(q, pad4)
            k = tf.pad(k, pad4)
            v = tf.pad(v, pad4)
            log_decay = tf.pad(log_decay, pad4)
            beta = tf.pad(beta, pad4)
        chunks = (tokens + padding) // chunk
        batch = tf.shape(q)[0]

        def to_chunks(x, depth):
            x = tf.reshape(x, [batch, chunks, chunk, heads, depth])
            return tf.transpose(x, [0, 1, 3, 2, 4])

        q = to_chunks(q, key_dim)
        k = to_chunks(k, key_dim)
        v = to_chunks(v, value_dim)
        log_decay = to_chunks(log_decay, key_dim)
        beta = to_chunks(beta, 1)

        cumulative = tf.cumsum(log_decay, axis=3)
        final = cumulative[:, :, :, -1:, :]
        # Splitting the decay symmetrically about the chunk midpoint keeps both
        # factors of exp(cumulative_i - cumulative_j) inside the float32 range.
        half = 0.5 * final
        decayed_q = q * tf.exp(cumulative)
        decayed_k = k * tf.exp(cumulative)
        trailing_k = k * tf.exp(final - cumulative)
        centered = tf.exp(cumulative - half)
        uncentered = k * tf.exp(half - cumulative)

        rows = tf.range(chunk)[:, None]
        columns = tf.range(chunk)[None, :]
        zero = tf.constant(0.0, tf.float32)
        key_attn = tf.where(
            rows > columns,
            tf.einsum("bnhck,bnhdk->bnhcd", k * centered, uncentered), zero)
        query_attn = tf.where(
            rows >= columns,
            tf.einsum("bnhck,bnhdk->bnhcd", q * centered, uncentered), zero)

        # Invert the unit lower triangular (I + diag(beta) @ key_attn) with a
        # Neumann series; key_attn is strictly lower triangular, so it is
        # nilpotent and the series is exact after `chunk` terms.
        nilpotent = -beta * key_attn
        identity = tf.eye(chunk, dtype=tf.float32)
        inverse = identity + nilpotent
        power = nilpotent
        span = 2
        while span < chunk:
            power = tf.matmul(power, power)
            inverse = tf.matmul(inverse, identity + power)
            span *= 2

        beta_v = beta * v
        beta_k = beta * decayed_k
        final_decay = tf.transpose(tf.exp(final), [0, 1, 2, 4, 3])

        state = tf.zeros([batch, heads, key_dim, value_dim], tf.float32)
        outputs = []
        for index in range(chunks):
            delta = tf.matmul(
                inverse[:, index],
                beta_v[:, index] - tf.matmul(beta_k[:, index], state))
            outputs.append(tf.matmul(decayed_q[:, index], state) +
                           tf.matmul(query_attn[:, index], delta))
            state = state * final_decay[:, index] + tf.matmul(
                trailing_k[:, index], delta, transpose_a=True)

        outputs = tf.stack(outputs, axis=1) * tf.math.rsqrt(
            tf.cast(key_dim, tf.float32))
        outputs = tf.transpose(outputs, [0, 1, 3, 2, 4])
        outputs = tf.reshape(
            outputs, [batch, chunks * chunk, heads, value_dim])
        return outputs[:, :tokens]


class ApplyAttentionPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyAttentionPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(apm.apm_map)
        self.idx = tf.constant(apm.apm_out, dtype=tf.int32)

    def call(self, logits, pp_logits):
        logits = tf.concat([tf.reshape(logits, [-1, 64 * 64]),
                            tf.reshape(pp_logits, [-1, 8 * 24])],
                           axis=1)
        idx = tf.repeat(tf.reshape(self.idx, [1, -1]), tf.shape(logits)[0], axis=0)
        return tf.gather(logits, idx, batch_dims=1)
        return tf.matmul(logits, tf.cast(self.fc1, logits.dtype))





class Metric:
    def __init__(self, short_name, long_name, suffix="", **kwargs):
        self.short_name = short_name
        self.long_name = long_name
        self.suffix = suffix
        self.value = 0.0
        self.count = 0

    def assign(self, value):
        self.value = value
        self.count = 1

    def accumulate(self, value):
        if self.count > 0:
            self.value = self.value + value
            self.count = self.count + 1
        else:
            self.assign(value)

    def merge(self, other):
        assert self.short_name == other.short_name
        self.value = self.value + other.value
        self.count = self.count + other.count

    def get(self):
        if self.count == 0:
            return self.value
        return self.value / self.count

    def reset(self):
        self.value = 0.0
        self.count = 0


class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = Net()
        self.root_dir = os.path.join(self.cfg["training"]["path"],
                                     self.cfg["name"])

        # Thresholds for policy_threshold_accuracy
        self.accuracy_thresholds = self.cfg["training"].get(
            "accuracy_thresholds", [1, 2, 5, 10])

        # Sparse training
        self.sparse = self.cfg["training"].get("sparse", False)

        # Network structure
        self.embedding_size = self.cfg["model"]["embedding_size"]
        self.pol_embedding_size = self.cfg["model"].get(
            "policy_embedding_size", self.embedding_size)
        self.val_embedding_size = self.cfg["model"].get(
            "value_embedding_size", 32)
        self.mov_embedding_size = self.cfg["model"].get(
            "moves_left_embedding_size", 8)
        self.encoder_layers = self.cfg["model"]["encoder_layers"]
        self.encoder_heads = self.cfg["model"]["encoder_heads"]
        self.encoder_d_model = self.cfg["model"].get("encoder_d_model")
        self.encoder_mixer_pattern = self.cfg["model"].get(
            "encoder_mixer_pattern", ["mha"])
        self.kda_key_dim = self.cfg["model"].get("kda_key_dim", 32)
        self.kda_value_dim = self.cfg["model"].get("kda_value_dim", 32)
        self.kda_gate_rank = self.cfg["model"].get("kda_gate_rank", 32)
        # Best chunk size is backend-specific; 16 was measured fastest on
        # DirectML. Tune per backend with tests/bench_mixers.py.
        self.kda_chunk_size = self.cfg["model"].get(
            "kda_chunk_size", KDA_CHUNK_SIZE)
        self.kda_directions = self.cfg["model"].get(
            "kda_directions",
            ["rank_forward", "rank_reverse", "file_forward", "file_reverse",
             "diag_forward", "diag_reverse",
             "anti_diag_forward", "anti_diag_reverse"])
        self.kda_output_gate = self.cfg["model"].get(
            "kda_output_gate", True)

        valid_mixers = {"mha", "kda"}
        if (not isinstance(self.encoder_mixer_pattern, (list, tuple)) or
                not self.encoder_mixer_pattern):
            raise ValueError("encoder_mixer_pattern must be a non-empty list")
        unknown_mixers = set(self.encoder_mixer_pattern) - valid_mixers
        if unknown_mixers:
            raise ValueError("Unknown encoder mixer(s): {}".format(
                ", ".join(sorted(unknown_mixers))))
        self.encoder_mixers = [
            self.encoder_mixer_pattern[index % len(self.encoder_mixer_pattern)]
            for index in range(self.encoder_layers)
        ]

        if "kda" in self.encoder_mixers:
            valid_directions = {
                "rank_forward", "rank_reverse",
                "file_forward", "file_reverse",
                "diag_forward", "diag_reverse",
                "anti_diag_forward", "anti_diag_reverse",
            }
            if (not isinstance(self.kda_directions, (list, tuple)) or
                    not self.kda_directions):
                raise ValueError("kda_directions must be a non-empty list")
            unknown_directions = set(self.kda_directions) - valid_directions
            if unknown_directions:
                raise ValueError("Unknown KDA direction(s): {}".format(
                    ", ".join(sorted(unknown_directions))))
            if len(set(self.kda_directions)) != len(self.kda_directions):
                raise ValueError("kda_directions must not contain duplicates")
            if self.encoder_heads % len(self.kda_directions) != 0:
                raise ValueError(
                    "encoder_heads must be divisible by the number of KDA directions")
            if min(self.kda_key_dim, self.kda_value_dim,
                   self.kda_gate_rank) <= 0:
                raise ValueError("KDA dimensions must be positive")
        self.categorical_value_buckets = self.cfg["model"].get(
            "categorical_value_buckets", 0)

        self.encoder_dff = self.cfg["model"].get(
            "encoder_dff", (self.embedding_size*1.5)//1)
        self.policy_d_model = self.cfg["model"].get(
            "policy_d_model", self.embedding_size)
        self.dropout_rate = self.cfg["model"].get("dropout_rate", 0.0)

        precision = self.cfg["training"].get("precision", "single")
        loss_scale = self.cfg["training"].get("loss_scale", 128)
        # added as part of Nadam needs added pr, code is near line 317
        self.weight_decay = self.cfg["training"].get("weight_decay", 0.0)
        self.beta_1 = self.cfg["training"].get(
            "beta_1", 0.9)  # Nadam beta1 default is 0.9
        self.beta_2 = self.cfg["training"].get(
            "beta_2", 0.999)  # Nadam beta2 default is 0.999
        self.epsilon = self.cfg["training"].get(
            "epsilon", 1e-07)  # Nadam epsilon value
        self.virtual_batch_size = self.cfg["model"].get(
            "virtual_batch_size", None)
        self.optimizer_name = self.cfg["training"].get(
            "optimizer", "sgd").lower()

        self.soft_policy_temperature = self.cfg["model"].get(
            "soft_policy_temperature", 1.0)
        self.value_st_scalar_loss = self.cfg["model"].get(
            "value_st_scalar_loss", False)

        self.use_smolgen = self.cfg["model"].get("use_smolgen", False)
        self.use_logit_gating = self.cfg["model"].get("use_logit_gating", False)
        assert not (self.use_smolgen and self.use_logit_gating), "Cannot use both smolgen and logit gating"
        
        self.smolgen_hidden_channels = self.cfg["model"].get(
            "smolgen_hidden_channels")
        self.smolgen_hidden_sz = self.cfg["model"].get("smolgen_hidden_sz")
        self.smolgen_gen_sz = self.cfg["model"].get("smolgen_gen_sz")
        self.smolgen_activation = self.cfg["model"].get("smolgen_activation")

        self.skip_first_ln = self.cfg["model"].get("skip_first_ln", False)
        self.omit_qkv_biases =  self.cfg["model"].get("omit_qkv_biases", False)
        self.omit_other_biases = self.cfg["model"].get("omit_other_biases", False)

        self.embedding_style = self.cfg["model"].get(
            "embedding_style", "new").lower()
        
        self.return_attn_wts = self.cfg["model"].get("return_attn_wts", False)

        # experiments with changing have failed
        self.encoder_norm = tf.keras.layers.LayerNormalization

        if precision == "single":
            self.model_dtype = tf.float32
        elif precision == "half":
            self.model_dtype = tf.float16
        elif precision == "bfloat16":
            self.model_dtype = tf.bfloat16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow. bfloat16 keeps the
        # float32 exponent range, so it does not need scaling.
        self.loss_scale = loss_scale if self.model_dtype == tf.float16 else 1

        policy_head = self.cfg['model'].get('policy', 'attention')
        value_head = self.cfg['model'].get('value', 'wdl')
        moves_left_head = self.cfg['model'].get('moves_left', 'v1')
        input_mode = self.cfg['model'].get('input_type', 'classic')
        default_activation = self.cfg['model'].get('default_activation',
                                                   'mish')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None
        self.MOVES_LEFT_HEAD = None
        self.INPUT_MODE = None
        self.DEFAULT_ACTIVATION = None

        if policy_head == "classical":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CONVOLUTION
        elif policy_head == "attention":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_ATTENTION
            if self.encoder_layers > 0:
                self.net.set_pol_headcount(self.encoder_heads)
        else:
            raise ValueError(
                "Unknown policy head format: {}".format(policy_head))

        self.net.set_policyformat(self.POLICY_HEAD)

        if value_head == "classical":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_CLASSICAL
            self.wdl = False
        elif value_head == "wdl":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_WDL
            self.wdl = True
        else:
            raise ValueError(
                "Unknown value head format: {}".format(value_head))

        self.net.set_valueformat(self.VALUE_HEAD)

        if moves_left_head == "none":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_NONE
            self.moves_left = False
        elif moves_left_head == "v1":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_V1
            self.moves_left = True
        else:
            raise ValueError(
                "Unknown moves left head format: {}".format(moves_left_head))

        self.net.set_movesleftformat(self.MOVES_LEFT_HEAD)

        if input_mode == "classic":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
        elif input_mode == "frc_castling":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
        elif input_mode == "canonical":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
        elif input_mode == "canonical_100":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
        elif input_mode == "canonical_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
        elif input_mode == "canonical_v2":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
        elif input_mode == "canonical_v2_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
        else:
            raise ValueError(
                "Unknown input mode format: {}".format(input_mode))

        self.net.set_input(self.INPUT_MODE)

        if default_activation == "relu":
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_RELU)
            self.DEFAULT_ACTIVATION = 'relu'
        elif default_activation == "mish":
            self.net.set_defaultactivation(
                pb.NetworkFormat.DEFAULT_ACTIVATION_MISH)
            try:
                self.DEFAULT_ACTIVATION = tf.keras.activations.mish
            except:
                import tensorflow_addons as tfa
                self.DEFAULT_ACTIVATION = tfa.activations.mish
                
        else:
            raise ValueError("Unknown default activation type: {}".format(
                default_activation))

        if self.encoder_layers > 0:
            self.net.set_headcount(self.encoder_heads)
            if "kda" in self.encoder_mixers:
                self.net.set_networkformat(
                    pb.NetworkFormat.NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT)
                self.net.set_kda_directions(self.kda_directions)
            else:
                self.net.set_networkformat(
                    pb.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT)
            for index, mixer in enumerate(self.encoder_mixers):
                self.net.set_encoder_mixer(
                    index, mixer,
                    key_dim=self.kda_key_dim,
                    value_dim=self.kda_value_dim,
                    gate_rank=self.kda_gate_rank,
                    output_gate=self.kda_output_gate)
            self.net.set_smolgen_activation(
                self.net.activation(self.smolgen_activation))
            self.net.set_ffn_activation(self.net.activation(
                'default'))

        if self.embedding_style == "new":
            self.net.set_input_embedding(
                pb.NetworkFormat.INPUT_EMBEDDING_PE_DENSE)
        elif self.encoder_layers > 0:
            self.net.set_input_embedding(
                pb.NetworkFormat.INPUT_EMBEDDING_PE_MAP)
        else:
            self.net.set_input_embedding(
                pb.NetworkFormat.INPUT_EMBEDDING_NONE)

        self.ffn_activation = self.cfg["model"].get(
            "ffn_activation", self.DEFAULT_ACTIVATION)

        assert default_activation == "mish" and self.cfg["model"].get(
            "ffn_activation") in [None, 'mish'], "Only mish is supported for now"

        self.swa_enabled = self.cfg["training"].get("swa", False)

        self.embedding_dense_sz = self.cfg["model"].get(
            "embedding_dense_sz", 128)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg["training"].get("swa_max_n", 0)

        self.renorm_enabled = self.cfg["training"].get("renorm", False)
        self.renorm_max_r = self.cfg["training"].get("renorm_max_r", 1)
        self.renorm_max_d = self.cfg["training"].get("renorm_max_d", 0)
        self.renorm_momentum = self.cfg["training"].get(
            "renorm_momentum", 0.99)

        device_type = accelerator_device_type()
        gpus = tf.config.experimental.list_physical_devices(device_type)
        if self.cfg['gpu'] == 'all':
            for gpu in gpus:
                enable_memory_growth(gpu)
            active_gpus = [f"{device_type}:{i}" for i in range(len(gpus))]
            self.strategy = tf.distribute.MirroredStrategy(active_gpus or None)
            tf.distribute.experimental_set_strategy(self.strategy)
        elif "," in str(self.cfg['gpu']):
            active_gpus = []
            for gpu in gpus:
                enable_memory_growth(gpu)
            for i in self.cfg['gpu'].split(","):
                active_gpus.append(f"{device_type}:{i}")
            self.strategy = tf.distribute.MirroredStrategy(active_gpus)
            tf.distribute.experimental_set_strategy(self.strategy)
        else:
            print(gpus)
            tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                       device_type)
            enable_memory_growth(gpus[self.cfg['gpu']])
            self.strategy = None
        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        elif self.model_dtype == tf.bfloat16:
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False,
                                       dtype=tf.int64)

    def init(self, train_dataset, test_dataset, validation_dataset=None):
        if self.strategy is not None:
            self.train_dataset = self.strategy.experimental_distribute_dataset(
                train_dataset)
        else:
            self.train_dataset = train_dataset
        self.train_iter = iter(self.train_dataset)
        if self.strategy is not None:
            self.test_dataset = self.strategy.experimental_distribute_dataset(
                test_dataset)
        else:
            self.test_dataset = test_dataset
        self.test_iter = iter(self.test_dataset)
        if self.strategy is not None and validation_dataset is not None:
            self.validation_dataset = self.strategy.experimental_distribute_dataset(
                validation_dataset)
        else:
            self.validation_dataset = validation_dataset
        if self.strategy is not None:
            this = self
            with self.strategy.scope():
                this.init_net()
        else:
            self.init_net()

    def init_net(self):
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        input_var = tf.keras.Input(shape=(112, 8, 8))
        outputs = self.construct_net(input_var)
        self.model = tf.keras.Model(inputs=input_var, outputs=outputs)

        

        print(f"params: {self.model.count_params()}")
        smolgen_params = np.sum([K.count_params(w) for w in self.model.trainable_weights if "smolgen" in w.name])
        emb_params = np.sum([K.count_params(w) for w in self.model.trainable_weights if "embedding/preprocess" in w.name])

        
        print(f"smolgen params: {smolgen_params}")
        print(f"emb preproc params: {emb_params}")



        try:
            flops = tfm.core.train_utils.try_count_flops(self.model)
            print(f"FLOPS: {flops / 10 ** 9:.03} G")
        except (AttributeError, ImportError):
            print("keras flops module not found, won't count flops")


        # swa_count initialized regardless to make checkpoint code simpler.
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
        self.swa_weights = None
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_weights = [
                tf.Variable(w, trainable=False) for w in self.model.weights
            ]

        self.active_lr = tf.Variable(0.000001, trainable=False)
        # All 'new' (TF 2.10 or newer non-legacy) optimizers must have learning_rate updated manually.
        self.update_lr_manually = True
        # Be sure not to set new_optimizer before TF 2.11, or unless you edit the code to specify a new optimizer explicitly.
        if self.optimizer_name == "sgd":
            if self.cfg['training'].get('new_optimizer'):
                self.optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.active_lr, momentum=0.9, nesterov=True)
                self.update_lr_manually = True
            else:
                try:
                    self.optimizer = tf.keras.optimizers.legacy.SGD(
                        learning_rate=lambda: self.active_lr,
                        momentum=0.9,
                        nesterov=True)
                except AttributeError:
                    self.optimizer = tf.keras.optimizers.SGD(
                        learning_rate=lambda: self.active_lr,
                        momentum=0.9,
                        nesterov=True)
        elif self.optimizer_name == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.active_lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=True)
        elif self.optimizer_name == "nadam":
            self.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=self.active_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
        else:
            raise ValueError("Unknown optimizer: " + self.optimizer_name)

        self.orig_optimizer = self.optimizer
        try:
            self.aggregator = self.orig_optimizer.aggregate_gradients
        except AttributeError:
            self.aggregator = self.orig_optimizer.gradient_aggregator
        if self.loss_scale != 1:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                self.optimizer, dynamic=True)

        def split_value_buckets(x, n_buckets=None, lo=-1, hi=1):
            if n_buckets is None:
                n_buckets = self.categorical_value_buckets
            x = tf.clip_by_value(x, lo, hi - 1e-9)
            x = (x - lo) / (hi - lo) * n_buckets
            x = tf.cast(x, tf.int32)
            return tf.one_hot(x, n_buckets, dtype=tf.float32)

        def categorical_value_loss(target, output):
            target = convert_val_to_scalar(target, softmax=False)
            target = split_value_buckets(target)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            return tf.reduce_mean(loss)

        def correct_policy(target, output, temperature=1.0):
            # Calculate loss on policy head
            if self.cfg["training"].get("mask_legal_moves"):
                # extract mask for legal moves from target policy
                move_is_legal = tf.greater_equal(target, 0)
                # replace logits of illegal moves with large negative value (so that it doesn"t affect policy of legal moves) without gradient
                illegal_filler = tf.zeros_like(output) - 1.0e10
                output = tf.where(move_is_legal, output, illegal_filler)
            # y_ still has -1 on illegal moves, flush them to 0
            target = tf.pow(tf.nn.relu(target), 1.0 / temperature)
            # normalize
            target = target / \
                tf.reduce_sum(input_tensor=target, axis=1, keepdims=True)
            return target, output

        def policy_loss(target, output, weights=None, temperature=1.0):
            if target.dtype == tf.int32:
                target = tf.one_hot(target, 1858)
                weights = tf.reduce_sum(target, axis=1, keepdims=False)
                target = target + (1 - tf.reduce_sum(target, axis=1, keepdims=True)) * (
                    1.0 / 1858)

            else:
                target, output = correct_policy(target, output, temperature)
            policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            target_entropy = tf.math.negative(
                tf.reduce_sum(tf.math.xlogy(target, target), axis=1))

            policy_kld = policy_cross_entropy - target_entropy
            if weights != None:
                policy_kld *= weights
            return tf.reduce_mean(policy_kld)

        self.policy_loss_fn = policy_loss

        def future_loss(target, output, opponent=False):
            target = tf.one_hot(target, 1858)
            target_u, target_d = apm.get_up_down(target)
            if opponent:
                target_u = tf.reverse(target_u, axis=[-1])
                target_d = tf.reverse(target_d, axis=[-1])
            output_u, output_d = tf.split(output, 2, axis=-1)
            output_u, output_d = tf.squeeze(output_u), tf.squeeze(output_d)
            print(output_u.shape, output_d.shape, target_u.shape, target_d.shape)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target_u), logits=output_u)
                + tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target_d), logits=output_d))
            return loss

        self.future_loss_fn = future_loss
        


        def policy_divergence(y1, y2, target):
            _, y1 = correct_policy(target, y1)
            y1 = tf.nn.softmax(y1)
            _, y2 = correct_policy(target, y2)
            policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(y1), logits=y2)
            y1_entropy = tf.math.negative(
                tf.reduce_sum(tf.math.xlogy(y1, y1), axis=1))
            policy_kld = policy_cross_entropy - y1_entropy
            return tf.reduce_mean(input_tensor=policy_kld)

        self.policy_divergence_fn = policy_divergence

        def get_policy_optimism_weights(value_target, value_pred, value_err_pred, strength=2.0):
            # value err pred already has square root taken
            value_pred = self.convert_val_to_scalar(value_pred, softmax=True)
            value_target = self.convert_val_to_scalar(value_target)
            value_err_pred = tf.math.sqrt(value_err_pred)
            z_values = (value_target - value_pred) / (value_err_pred + 1e-5)
            weights = tf.math.sigmoid((z_values - strength) * 3)
            weights = tf.stop_gradient(weights)
            weights = tf.reshape(weights, [-1])
            return weights

        self.policy_optimism_weights_fn = get_policy_optimism_weights

        def policy_accuracy(target, output):
            target, output = correct_policy(target, output)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.policy_accuracy_fn = policy_accuracy

        def moves_left_mean_error_fn(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(tf.abs(target - output))

        self.moves_left_mean_error = moves_left_mean_error_fn

        def policy_entropy(target, output):
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            return tf.math.negative(
                tf.reduce_mean(
                    tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                                  axis=1)))

        self.policy_entropy_fn = policy_entropy

        def policy_uniform_loss(target, output):
            uniform = tf.where(tf.greater_equal(target, 0),
                               tf.ones_like(target), tf.zeros_like(target))
            balanced_uniform = uniform / tf.reduce_sum(
                uniform, axis=1, keepdims=True)
            target, output = correct_policy(target, output)
            policy_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(balanced_uniform),
                                                        logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_uniform_loss_fn = policy_uniform_loss

        def policy_search_loss(target, output, epsilon=0.003):
            # output and target both have shape (batch_size, num_outputs)
            # time to search is roughly 1 / [prediction at best move]
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            best_moves = tf.argmax(input=target, axis=1, output_type=tf.int32)
            # output at the best_moves locations
            output_at_best_moves = tf.gather_nd(softmaxed, tf.stack(
                [tf.range(tf.shape(output)[0]), best_moves], axis=1))

            # estimated search time
            search_time = 1.0 / (output_at_best_moves + epsilon)
            return tf.reduce_mean(search_time)

        self.policy_search_loss_fn = policy_search_loss

        def policy_thresholded_accuracy(target, output, thresholds=None):
            # thresholds can be a list of thresholds or a single threshold
            # if no threshold argument defaults to self.accuracy_thresholds
            # Rate at which the best move has policy > threshold%
            if thresholds is None:
                thresholds = self.accuracy_thresholds
            if isinstance(thresholds, float):
                thresholds = [thresholds]
            if not thresholds:
                return []
            thresholds = [threshold / 100 for threshold in thresholds]
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            best_moves = tf.argmax(input=target, axis=1, output_type=tf.int32)
            # output at the best_moves locations
            output_at_best_moves = tf.gather_nd(softmaxed, tf.stack(
                [tf.range(tf.shape(output)[0]), best_moves], axis=1))
            accuracies = []
            for threshold in thresholds:
                accuracy = tf.cast(tf.greater(
                    output_at_best_moves, threshold), tf.float32)
                accuracies.append(tf.reduce_mean(accuracy))
            return accuracies

        self.policy_thresholded_accuracy_fn = policy_thresholded_accuracy

        # Linear conversion to scalar to compute MSE with, for comparison to old values
        wdl = tf.expand_dims(tf.constant([1.0, 0.0, -1.0]), 1)

        def unreduced_mse_loss(target, output):
            assert target.shape[-1] == output.shape[-1]
            scalar_z_conv = convert_val_to_scalar(output, softmax=True)
            scalar_target = convert_val_to_scalar(target, softmax=False)
            return tf.math.squared_difference(
                scalar_target, scalar_z_conv)

        def convert_val_to_scalar(val, softmax=False):
            if val.shape[-1] == 3:
                if softmax:
                    val = tf.nn.softmax(val)
                return tf.matmul(val, wdl)
            elif val.shape[-1] == 1:
                return val
            else:
                raise ValueError(
                    "Value head size must be 3 or 1 but got: {}".format(val.shape[-1]))

        self.convert_val_to_scalar = convert_val_to_scalar

        def mse_loss(target, output):
            return tf.reduce_mean(input_tensor=unreduced_mse_loss(target, output))

        self.mse_loss_fn = mse_loss

        def value_loss(target, output):
            output = tf.cast(output, tf.float32)
            if output.shape[-1] == 1:
                return mse_loss(target, output)
            else:
                # subtract target entropy for consistency across value heads
                value_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(target), logits=output)
                return tf.reduce_mean(input_tensor=value_cross_entropy)

        self.value_loss_fn = value_loss

        def value_err_loss(value_target, value, output):
            value = convert_val_to_scalar(value, softmax=True)
            value_target = convert_val_to_scalar(value_target, softmax=False)
            true_error = tf.stop_gradient(tf.math.squared_difference(value_target, value))
            loss = tf.math.squared_difference(true_error, output)
            return tf.reduce_mean(input_tensor=loss)

        self.value_err_loss_fn = value_err_loss

        def value_losses(target, output, err_output=None, cat_output=None,
                         scalar_primary=False):
            value_loss = (self.mse_loss_fn(target, output)
                          if scalar_primary else self.value_loss_fn(target, output))
            value_err_loss = self.value_err_loss_fn(target, tf.stop_gradient(
                output), err_output) if err_output is not None else tf.constant(0.)
            value_cat_loss = categorical_value_loss(
                target, cat_output) if cat_output is not None else tf.constant(0.)

            return value_loss, value_err_loss, value_cat_loss

        self.value_losses_fn = value_losses

        if self.moves_left:

            def moves_left_loss(target, output):
                # Scale the loss to similar range as other losses.
                scale = 20.0
                target = target / scale
                output = tf.cast(output, tf.float32) / scale
                if self.strategy is not None:
                    huber = tf.keras.losses.Huber(
                        10.0 / scale, reduction=tf.keras.losses.Reduction.NONE)
                else:
                    huber = tf.keras.losses.Huber(10.0 / scale)
                return tf.reduce_mean(huber(target, output))
        else:
            moves_left_loss = None

        self.moves_left_loss_fn = moves_left_loss
        self.possible_losses = ["policy",
                                "policy_optimistic_st",
                                "policy_soft",
                                "policy_opponent",
                                "policy_next",
                                "value_winner",
                                "value_q",
                                "value_q_err",
                                "value_q_cat",
                                "value_st",
                                "value_st_err",
                                "value_st_cat",
                                "reg",
                                "moves_left",
                                ]
        self.loss_weights = self.cfg["training"]["loss_weights"]
        for key in self.loss_weights:
            if key not in self.possible_losses:
                print(key, self.possible_losses)
                raise ValueError("Unrecognized loss: {}".format(key))

        def _lossMix(losses):
            for key in losses:
                assert key in self.possible_losses, "Unrecognized loss: {}".format(
                    key)
                losses[key] = losses[key] * \
                    self.loss_weights.get(key, tf.constant(0.0))
            return sum(losses.values())

        self.lossMix = _lossMix

        def accuracy(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.accuracy_fn = accuracy

        accuracy_thresholded_metrics = []
        for threshold in self.accuracy_thresholds:
            accuracy_thresholded_metrics.append(
                Metric(f"P@{threshold}%", f"Thresholded Policy Accuracy @ {threshold}"))

        # Order must match the order in process_inner_loop

        self.train_metrics = [
            Metric("P", "Policy Loss"),
            Metric("POST", "Policy Optimistic ST Loss"),
            Metric("POST KLD", "Policy Optimistic KLD"),
            Metric("SP", "Soft Policy Loss"),
            Metric("ML", "Moves Left Loss"),
            Metric("Reg", "Reg term"),
            Metric("Total", "Total Loss"),
            Metric(
                "V MSE", "MSE Loss"
            ),  # Long name here doesn"t mention value for backwards compatibility reasons.
            Metric("P Acc", "Policy Accuracy", suffix="%"),
            Metric("V Acc", "Value Accuracy", suffix="%"),
            Metric("P Entropy", "Policy Entropy"),
            Metric("P UL", "Policy UL"),
            Metric("P SL", "Policy SL"),
            Metric("VW", "Value Winner Loss"),
            Metric("VQ", "Value Q Loss"),
            Metric("V Q Err", "Value Err L"),
            Metric("V Q Cat", "Value Cat L"),
            Metric("V ST", "Value ST Loss"),
            Metric("V ST Err", "Value ST Err Loss"),
            Metric("V ST Cat", "Value ST Cat Loss"),
            Metric("P Opp", "Policy Opponent Loss"),
            Metric("P Next", "Policy Next Loss"),
        ]

        self.train_metrics.extend(accuracy_thresholded_metrics)
        self.time_start = None
        self.last_steps = None

        # Order must match the order in calculate_test_summaries_inner_loop
        self.test_metrics = [
            Metric("P", "Policy Loss"),
            Metric("POST", "Policy Optimistic ST Loss"),
            Metric("POST KLD", "Policy Optimistic KLD"),
            Metric("SP", "Soft Policy Loss"),
            Metric("ML", "Moves Left Loss"),
            Metric(
                "V MSE", "MSE Loss"
            ),  # Long name here doesn"t mention value for backwards compatibility reasons.
            Metric("P Acc", "Policy Accuracy", suffix="%"),
            Metric("V Acc", "Value Accuracy", suffix="%"),
            Metric("P Entropy", "Policy Entropy"),
            Metric("P UL", "Policy UL"),
            Metric("P SL", "Policy SL"),
            Metric("VW", "Value Winner Loss"),
            Metric("VQ", "Value Q Loss"),
            Metric("V Q Err", "Value Err L"),
            Metric("V Q Cat", "Value Cat L"),
            Metric("V ST", "Value ST Loss"),
            Metric("V ST Err", "Value ST Err Loss"),
            Metric("V ST Cat", "Value ST Cat Loss"),
            Metric("P Opp", "Policy Opponent Loss"),
            Metric("P Next", "Policy Next Loss"),
        ]

        self.test_metrics.extend(accuracy_thresholded_metrics)

        self.detailed_summaries = self.cfg["training"].get(
            "detailed_summaries", True)

        # Set adaptive learning rate during training
        self.cfg["training"]["lr_boundaries"].sort()
        self.warmup_steps = self.cfg["training"].get("warmup_steps", 0)
        self.lr = self.cfg["training"]["lr_values"][0]
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-test".format(self.cfg["name"])))
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-train".format(self.cfg["name"])))
        if vars(self).get("validation_dataset", None) is not None:
            self.validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-validation".format(self.cfg["name"])))
        if self.swa_enabled:
            self.swa_writer = tf.summary.create_file_writer(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-swa-test".format(self.cfg["name"])))
            self.swa_validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-swa-validation".format(self.cfg["name"])))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer,
                                              model=self.model,
                                              global_step=self.global_step,
                                              swa_count=self.swa_count)
        self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.root_dir,
            max_to_keep=50,
            keep_checkpoint_every_n_hours=24,
            checkpoint_name=self.cfg["name"])

    # False to True is a hack to keep net to model working with atnb
    def replace_weights(self, proto_filename: str, ignore_errors: bool = False):
        self.net.parse_proto(proto_filename)

        filters, blocks = self.net.filters(), self.net.blocks()
        if not ignore_errors:
            if self.POLICY_HEAD != self.net.pb.format.network_format.policy:
                raise ValueError("Policy head type doesn't match the network")
            if self.VALUE_HEAD != self.net.pb.format.network_format.value:
                raise ValueError("Value head type doesn't match the network")

        # List all tensor names we need weights for.
        names = []
        for weight in self.model.weights:
            names.append(weight.name)

        new_weights = self.net.get_weights_v2(names)
        for weight in self.model.weights:
            if "renorm" in weight.name:
                # Renorm variables are not populated.
                continue

            try:
                new_weight = new_weights[weight.name]
            except KeyError:
                error_string = "No values for tensor {} in protobuf".format(
                    weight.name)
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if reduce(operator.mul, weight.shape.as_list(),
                      1) != len(new_weight):
                error_string = "Tensor {} has wrong length. Tensorflow shape {}, size in protobuf {}".format(
                    weight.name, weight.shape.as_list(), len(new_weight))
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if weight.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if weight.name == "input/conv2d/kernel:0" and self.net.pb.format.network_format.input < pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weight)):
                        if (i % (num_inputs * 9)) // 9 == rule50_input:
                            new_weight[i] = new_weight[i] * 99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weight.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weight.shape.ndims == 2:
                if ('/kda/a_log:' in weight.name or
                        '/kda/dt_bias:' in weight.name):
                    new_weight = tf.constant(
                        new_weight, shape=weight.shape)
                    weight.assign(new_weight)
                else:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    s = weight.shape.as_list()
                    shape = [s[i] for i in [1, 0]]
                    new_weight = tf.constant(new_weight, shape=shape)
                    weight.assign(tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weight, shape=weight.shape)
                weight.assign(new_weight)
        # Replace the SWA weights as well, ensuring swa accumulation is reset.
        if self.swa_enabled:
            self.swa_count.assign(tf.constant(0.))
            self.update_swa()
        # This should result in identical file to the starting one
        # self.save_leelaz_weights("restored.pb.gz")

    def restore(self):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            self.checkpoint.restore(self.manager.latest_checkpoint)

    def process_loop(self, batch_size: int, test_batches: int, batch_splits: int = 1):
        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2
        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_sizes * self.cfg[
                "training"].get("num_batch_splits", 1)
            raise ValueError(
                "batch_size must be a multiple of {}".format(required_factor))

        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg["training"]["total_steps"]

        def loop():
            for _ in range(steps % total_steps, total_steps):
                while os.path.exists("stop"):
                    time.sleep(1)
                self.process(batch_size, test_batches,
                             batch_splits=batch_splits)

        from importlib.util import find_spec
        if find_spec("rich") is not None:
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
            from rich.table import Column

            self.progressbar = Progress(
                BarColumn(),
                "[progress.percentage]{task.percentage:>4.2f}%",
                TimeRemainingColumn(),
                TextColumn("{task.completed:.2f} of {task.total} steps completed.",
                           table_column=Column(ratio=1)),
                # TextColumn("Policy accuracy {task.train_metrics[6].get():.2f}", table_column=Column(ratio=1)),
                SpinnerColumn(),
            )
            with self.progressbar:
                self.progresstask = self.progressbar.add_task(
                    f"[green]Doing {total_steps} training steps", total=total_steps)
                loop()
        else:
            print("Warning, rich module not found, disabling progress bar")
            loop()

    @tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    @tf.function()
    def process_inner_loop(self, x, y, z, q, m, st_q, opp_idx, next_idx):

        with tf.GradientTape() as tape:

            outputs = self.model(x, training=True)
            value_winner = outputs.get("value_winner")
            value_winner_err = None
            value_q = outputs.get("value_q")
            value_q_err = outputs.get("value_q_err")
            value_q_cat = outputs.get("value_q_cat")
            value_st = outputs.get("value_st")
            value_st_err = outputs.get("value_st_err")
            value_st_cat = outputs.get("value_st_cat")

            policy = outputs["policy"]
            policy_optimistic_st = outputs.get("policy_optimistic_st")
            policy_soft = outputs.get("policy_soft")

            policy_opponent = outputs.get("policy_opponent")
            policy_next = outputs.get("policy_next")

            # Policy losses
            policy_loss = self.policy_loss_fn(y, policy)
            policy_accuracy = self.policy_accuracy_fn(y, policy)
            policy_entropy = self.policy_entropy_fn(y, policy)
            policy_ul = self.policy_uniform_loss_fn(y, policy)
            policy_sl = self.policy_search_loss_fn(y, policy)
            policy_thresholded_accuracies = self.policy_thresholded_accuracy_fn(
                y, policy)
            if policy_optimistic_st is not None:
                optimism_weights = self.policy_optimism_weights_fn(
                    st_q, value_st, value_st_err)
                policy_optimistic_st_loss = self.policy_loss_fn(
                    y, policy_optimistic_st, weights=optimism_weights)
                policy_optimistic_st_divergence = self.policy_divergence_fn(
                    policy, policy_optimistic_st, y)
            else:
                policy_optimistic_st_loss = tf.constant(0.)
                policy_optimistic_st_divergence = tf.constant(0.)
            if policy_soft is not None:
                policy_soft_loss = self.policy_loss_fn(
                    y, policy_soft, temperature=self.soft_policy_temperature)
            else:
                policy_soft_loss = tf.constant(0.)

            policy_opponent_loss = self.future_loss_fn(
                opp_idx, policy_opponent, opponent=True) if policy_opponent is not None else tf.constant(0.)
            policy_next_loss = self.future_loss_fn(
                next_idx, policy_next) if policy_next is not None else tf.constant(0.)

            # Value losses
            value_winner_loss, value_winner_err_loss, value_winner_cat_loss = self.value_losses_fn(
                z, value_winner, None, None)
            value_q_loss, value_q_err_loss, value_q_cat_loss = self.value_losses_fn(
                q, value_q, value_q_err, value_q_cat) if value_q is not None else (tf.constant(0.), tf.constant(0.), tf.constant(0.))
            value_st_loss, value_st_err_loss, value_st_cat_loss = self.value_losses_fn(
                st_q, value_st, value_st_err, value_st_cat,
                scalar_primary=self.value_st_scalar_loss) if value_st is not None else (tf.constant(0.), tf.constant(0.), tf.constant(0.))
            if self.wdl:
                mse_loss = self.mse_loss_fn(q, value_q)
                value_accuracy = self.accuracy_fn(q, value_q)
            else:
                mse_loss = self.mse_loss_fn(q, value_q)
                value_accuracy = tf.constant(0.)

            reg_term = sum(self.model.losses)
            if self.moves_left:
                moves_left = outputs["moves_left"]
                moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            else:
                moves_left_loss = tf.constant(0.)

            losses = {
                "policy": policy_loss,
                "policy_optimistic_st": policy_optimistic_st_loss,
                "policy_soft": policy_soft_loss,
                "policy_opponent": policy_opponent_loss,
                "policy_next": policy_next_loss,
                "value_winner": value_winner_loss,
                "value_q": value_q_loss,
                "value_q_err": value_q_err_loss,
                "value_q_cat": value_q_cat_loss,
                "value_st": value_st_loss,
                "value_st_err": value_st_err_loss,
                "value_st_cat": value_st_cat_loss,
                "moves_left": moves_left_loss,
                "reg": reg_term,
            }

            total_loss = self.lossMix(losses)

            metrics = [
                policy_loss,
                policy_optimistic_st_loss,
                policy_optimistic_st_divergence,
                policy_soft_loss,
                moves_left_loss,
                reg_term,
                total_loss,
                # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
                # get comparable values.
                mse_loss / 4.0,
                policy_accuracy * 100,
                value_accuracy * 100,
                policy_entropy,
                policy_ul,
                policy_sl,
                value_winner_loss,
                value_q_loss,
                value_q_err_loss,
                value_q_cat_loss,
                value_st_loss,
                value_st_err_loss,
                value_st_cat_loss,
                policy_opponent_loss,
                policy_next_loss,
            ]
            metrics.extend(
                [acc * 100 for acc in policy_thresholded_accuracies])

            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)

        return metrics, tape.gradient(total_loss, self.model.trainable_weights)

    @tf.function()
    def strategy_process_inner_loop(self, x, y, z, q, m, st_q, opp_idx, next_idx):
        metrics, new_grads = self.strategy.run(self.process_inner_loop,
                                               args=(x, y, z, q, m, st_q, opp_idx, next_idx))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics, new_grads

    @tf.function()
    def apply_grads(self, grads, effective_batch_splits):
        grads = [
            g[0]
            for g in self.aggregator(zip(grads, self.model.trainable_weights))
        ]
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['training'].get(
            'max_grad_norm', 10000.0) * effective_batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights),
                                       experimental_aggregate_gradients=False)
        return grad_norm

    @tf.function()
    def strategy_apply_grads(self, grads, effective_batch_splits: int):
        grad_norm = self.strategy.run(self.apply_grads,
                                      args=(grads, effective_batch_splits))
        grad_norm = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         grad_norm,
                                         axis=None)
        return grad_norm

    @tf.function()
    def merge_grads(self, grads, new_grads):
        return [tf.math.add(a, b) for (a, b) in zip(grads, new_grads)]

    @tf.function()
    def strategy_merge_grads(self, grads, new_grads):
        return self.strategy.run(self.merge_grads, args=(grads, new_grads))


    def train_step(self, steps: int, batch_size: int, batch_splits: int):
        # need to add 1 to steps because steps will be incremented after gradient update
        if self.detailed_summaries and ((steps +
                1) % self.cfg["training"]["train_avg_report_steps"] == 0 or (
                steps + 1) % self.cfg["training"]["total_steps"] == 0):
            before_weights = self.read_weights()

        # Run training for this batch
        grads = None
        # Read once: `steps` is a device tensor and syncing it per split stalls
        # the pipeline.
        completed = steps.numpy().item() - 1 if hasattr(
            self, "progressbar") else 0
        for batch_id in range(batch_splits):
            x, y, z, q, m, st_q, opp_idx, next_idx = next(self.train_iter)
            if self.strategy is not None:
                metrics, new_grads = self.strategy_process_inner_loop(
                    x, y, z, q, m, st_q, opp_idx, next_idx)
            else:
                metrics, new_grads = self.process_inner_loop(
                    x, y, z, q, m, st_q, opp_idx, next_idx)
            if not grads:
                grads = new_grads
            else:
                if self.strategy is not None:
                    grads = self.strategy_merge_grads(grads, new_grads)
                else:
                    grads = self.merge_grads(grads, new_grads)
            # Keep running averages
            for acc, val in zip(self.train_metrics, metrics):
                acc.accumulate(val)

            if hasattr(self, "progressbar"):
                self.progressbar.update(
                    self.progresstask,
                    completed=completed + (batch_id + 1) / batch_splits)
        # Gradients of batch splits are summed, not averaged like usual, so need to scale lr accordingly to correct for this.
        effective_batch_splits = batch_splits
        if self.strategy is not None:
            effective_batch_splits = batch_splits * self.strategy.num_replicas_in_sync
        self.active_lr.assign(self.lr / effective_batch_splits)
        if self.update_lr_manually:
            self.orig_optimizer.learning_rate = self.active_lr
        if self.strategy is not None:
            grad_norm = self.strategy_apply_grads(grads,
                                                  effective_batch_splits)
        else:
            grad_norm = self.apply_grads(grads, effective_batch_splits)

        # Note: grads variable at this point has not been unscaled or
        # had clipping applied. Since no code after this point depends
        # upon that it seems fine for now.

        # Update steps.
        self.global_step.assign_add(1)
        steps = self.global_step.read_value()

        if steps % self.cfg["training"][
                "train_avg_report_steps"] == 0 or steps % self.cfg["training"][
                    "total_steps"] == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (tf.cast(steps_elapsed, tf.float32) /
                                      elapsed)
            print("step {}, lr={:g}".format(steps, self.lr), end="")
            for metric in self.train_metrics:
                try:
                    print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                              metric.suffix),
                          end="")
                except:
                    print("failure to print metric", metric.short_name,
                          metric.get(), metric.suffix)
            print(" ({:g} pos/s)".format(speed))

            with self.train_writer.as_default():
                for metric in self.train_metrics:
                    tf.summary.scalar(metric.long_name,
                                      metric.get(),
                                      step=steps)
                tf.summary.scalar("LR", self.lr, step=steps)
                tf.summary.scalar("Gradient norm",
                                  grad_norm / effective_batch_splits,
                                  step=steps)
                if self.detailed_summaries:
                    after_weights = self.read_weights()
                    self.compute_update_ratio(
                        before_weights, after_weights, steps)
            self.train_writer.flush()

            self.time_start = time_end
            self.last_steps = steps
            for metric in self.train_metrics:
                metric.reset()

        if self.sparse:  # !!!
            if not hasattr(self, "sparsity_patterns"):
                self.set_sparsity_patterns()
            self.apply_sparsity()

        return steps

    def process(self, batch_size: int, test_batches: int, batch_splits: int):
        # Get the initial steps value before we do a training step.
        steps = self.global_step.read_value()

        # By default disabled since 0 != 10.
        if steps % self.cfg["training"].get("profile_step_freq",
                                            1) == self.cfg["training"].get(
                                                "profile_step_offset", 10):
            self.profiling_start_step = steps
            tf.profiler.experimental.start(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-profile".format(self.cfg["name"])))

        # Run test before first step to see delta since end of last run.
        if steps % self.cfg["training"]["total_steps"] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps + 1):
                # Steps is given as one higher than current in order to avoid it
                # being equal to the value the end of a run is stored against.
                self.calculate_test_summaries(test_batches, steps + 1)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps + 1)

        # Determine learning rate
        lr_values = self.cfg["training"]["lr_values"]
        lr_boundaries = self.cfg["training"]["lr_boundaries"]
        steps_total = steps % self.cfg["training"]["total_steps"]
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1,
                                        tf.float32) / self.warmup_steps

        with tf.profiler.experimental.Trace("Train", step_num=steps):
            steps = self.train_step(steps, batch_size, batch_splits)

        if self.swa_enabled and steps % self.cfg["training"]["swa_steps"] == 0:
            self.update_swa()

        # Calculate test values every "test_steps", but also ensure there is
        # one at the final step so the delta to the first step can be calculated.
        if steps % self.cfg["training"]["test_steps"] == 0 or steps % self.cfg[
                "training"]["total_steps"] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps):
                self.calculate_test_summaries(test_batches, steps)
                if self.swa_enabled:
                    self.calculate_swa_summaries(test_batches, steps)

        if self.validation_dataset is not None and (
                steps % self.cfg["training"]["validation_steps"] == 0
                or steps % self.cfg["training"]["total_steps"] == 0):
            if self.swa_enabled:
                self.calculate_swa_validations(steps)
            else:
                self.calculate_test_validations(steps)

        # Save session and weights at end, and also optionally every "checkpoint_steps".
        if steps % self.cfg["training"]["total_steps"] == 0 or (
                "checkpoint_steps" in self.cfg["training"]
                and steps % self.cfg["training"]["checkpoint_steps"] == 0):
            if True:

                # Checkpoint the model weights.
                evaled_steps = steps.numpy()
                self.manager.save(checkpoint_number=evaled_steps)
                print("Model saved in file: {}".format(
                    self.manager.latest_checkpoint))

                if not self.cfg["training"].get(
                        "disable_saved_model_checkpointing", False):
                    # Save normal weights
                    tf.saved_model.save(self.model, os.path.join(
                        self.root_dir, self.cfg["name"]) + str(evaled_steps))

                    # Save swa weights
                    if self.swa_enabled:
                        backup = self.read_weights()
                        for (swa, w) in zip(self.swa_weights, self.model.weights):
                            w.assign(swa.read_value())
                        evaled_steps = steps.numpy()
                        tf.saved_model.save(self.model, os.path.join(
                            self.root_dir, self.cfg["name"]) + "-swa-" + str(evaled_steps))
                        for (old, w) in zip(backup, self.model.weights):
                            w.assign(old)

                if not self.cfg["training"].get("disable_pb_checkpointing"):  
                    path = os.path.join(self.root_dir, self.cfg["name"])
                    leela_path = path + "-" + str(evaled_steps)
                    swa_path = path + "-swa-" + str(evaled_steps)
                    self.net.pb.training_params.training_steps = evaled_steps
                    self.save_leelaz_weights(leela_path)
                    if self.swa_enabled:
                        self.save_swa_weights(swa_path)

        if self.profiling_start_step is not None and (
                steps >= self.profiling_start_step +
                self.cfg["training"].get("profile_step_count", 0)
                or steps % self.cfg["training"]["total_steps"] == 0):
            tf.profiler.experimental.stop()
            self.profiling_start_step = None

    def calculate_swa_summaries(self, test_batches: int, steps: int):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print("swa", end=" ")
        self.calculate_test_summaries(test_batches, steps)
        self.test_writer = true_test_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y, z, q, m, st_q, opp_idx, next_idx):
        outputs = self.model(x, training=False)

        value_winner = outputs.get("value_winner")
        value_winner_err = None
        value_q = outputs.get("value_q")
        value_q_err = outputs.get("value_q_err")
        value_q_cat = outputs.get("value_q_cat")
        value_st = outputs.get("value_st")
        value_st_err = outputs.get("value_st_err")
        value_st_cat = outputs.get("value_st_cat")

        policy = outputs["policy"]
        policy_optimistic_st = outputs.get("policy_optimistic_st")
        policy_soft = outputs.get("policy_soft")

        policy_opponent = outputs.get("policy_opponent")
        policy_next = outputs.get("policy_next")

        # Policy losses
        policy_loss = self.policy_loss_fn(y, policy)
        policy_accuracy = self.policy_accuracy_fn(y, policy)
        policy_entropy = self.policy_entropy_fn(y, policy)
        policy_ul = self.policy_uniform_loss_fn(y, policy)
        policy_sl = self.policy_search_loss_fn(y, policy)
        policy_thresholded_accuracies = self.policy_thresholded_accuracy_fn(
            y, policy)
        if policy_optimistic_st is not None:
            optimism_weights = self.policy_optimism_weights_fn(
                st_q, value_st, value_st_err)
            policy_optimistic_st_loss = self.policy_loss_fn(
                y, policy_optimistic_st, weights=optimism_weights)
            policy_optimistic_st_divergence = self.policy_divergence_fn(
                policy, policy_optimistic_st, y)
        else:
            policy_optimistic_st_loss = tf.constant(0.)
            policy_optimistic_st_divergence = tf.constant(0.)
        if policy_soft is not None:
            policy_soft_loss = self.policy_loss_fn(
                y, policy_soft, temperature=self.soft_policy_temperature)
        else:
            policy_soft_loss = tf.constant(0.)
        policy_opponent_loss = self.future_loss_fn(
            opp_idx, policy_opponent, opponent=True) if policy_opponent is not None else tf.constant(0.)
        policy_next_loss = self.future_loss_fn(
            next_idx, policy_next) if policy_next is not None else tf.constant(0.)

        # Value losses
        value_winner_loss, value_winner_err_loss, value_winner_cat_loss = self.value_losses_fn(
            z, value_winner, None, None)

        value_q_loss, value_q_err_loss, value_q_cat_loss = self.value_losses_fn(
            q, value_q, value_q_err, value_q_cat) if value_q is not None else (tf.constant(0.),) * 3

        value_st_loss, value_st_err_loss, value_st_cat_loss = self.value_losses_fn(
            st_q, value_st, value_st_err, value_st_cat,
            scalar_primary=self.value_st_scalar_loss) if value_st is not None else (tf.constant(0.),) * 3

        if self.wdl:
            mse_loss = self.mse_loss_fn(q, value_q)
            value_accuracy = self.accuracy_fn(q, value_q)
        else:
            mse_loss = self.mse_loss_fn(q, value_q)
            value_accuracy = tf.constant(0.)

        # Moves left loss
        if self.moves_left:
            moves_left = outputs["moves_left"]
            moves_left_loss = self.moves_left_loss_fn(m, moves_left)
        else:
            moves_left_loss = tf.constant(0.)

        metrics = [
            policy_loss,
            policy_optimistic_st_loss,
            policy_optimistic_st_divergence,
            policy_soft_loss,
            moves_left_loss,
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss / 4.0,
            policy_accuracy * 100,
            value_accuracy * 100,
            policy_entropy,
            policy_ul,
            policy_sl,
            value_winner_loss,
            value_q_loss,
            value_q_err_loss,
            value_q_cat_loss,
            value_st_loss,
            value_st_err_loss,
            value_st_cat_loss,
            policy_opponent_loss,
            policy_next_loss,
        ]

        metrics.extend([acc * 100 for acc in policy_thresholded_accuracies])
        return metrics

    @tf.function()
    def strategy_calculate_test_summaries_inner_loop(self, x, y, z, q, m, st_q, opp_idx, next_idx):
        metrics = self.strategy.run(self.calculate_test_summaries_inner_loop,
                                    args=(x, y, z, q, m, st_q, opp_idx, next_idx))
        metrics = [
            self.strategy.reduce(tf.distribute.ReduceOp.MEAN, m, axis=None)
            for m in metrics
        ]
        return metrics

    def calculate_test_summaries(self, test_batches: int, steps: int):
        for metric in self.test_metrics:
            metric.reset()
        for _ in range(0, test_batches):
            x, y, z, q, m, st_q, opp_idx, next_idx = next(self.test_iter)
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m, st_q, opp_idx, next_idx)
            else:
                metrics = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m, st_q, opp_idx, next_idx)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        self.net.pb.training_params.learning_rate = self.lr
        self.net.pb.training_params.mse_loss = self.test_metrics[3].get()
        self.net.pb.training_params.policy_loss = self.test_metrics[0].get()
        # TODO store value and value accuracy in pb
        self.net.pb.training_params.accuracy = self.test_metrics[4].get()
        with self.test_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
            if self.detailed_summaries:
                for w in self.model.weights:
                    tf.summary.histogram(w.name, w, step=steps)
            params = self.model.count_params() 
            smolgen_params = np.sum([K.count_params(w) for w in self.model.trainable_weights if "smol" in w.name])
            emb_params = np.sum([K.count_params(w) for w in self.model.trainable_weights if "embedding/preprocess" in w.name])
            flops = (tfm.core.train_utils.try_count_flops(self.model)
                     if tfm is not None else None)
            if steps == 1:
                tf.summary.text("Params", str(smolgen_params), step=steps)
                tf.summary.text("Smolgen params", str(smolgen_params), step=steps)
                tf.summary.text("Embedding params", str(emb_params), step=steps)
                if flops is not None:
                    tf.summary.text("FLOPS", str(flops), step=steps)


        self.test_writer.flush()

        print("step {},".format(steps), end="")
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                      metric.suffix),
                  end="")
        print()

    def calculate_swa_validations(self, steps: int):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_validation_writer, self.validation_writer = self.validation_writer, self.swa_validation_writer
        print("swa", end=" ")
        self.calculate_test_validations(steps)
        self.validation_writer = true_validation_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def calculate_test_validations(self, steps: int):
        print("logging test validations")
        for metric in self.test_metrics:
            metric.reset()
        for (x, y, z, q, m, st_q, opp_idx, next_idx) in self.validation_dataset:
            if self.strategy is not None:
                metrics = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m, st_q, opp_idx, next_idx)
            else:
                metrics = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m, st_q, opp_idx, next_idx)
            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)
        with self.validation_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
        self.validation_writer.flush()

        print("step {}, validation:".format(steps), end="")
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(),
                                      metric.suffix),
                  end="")
        print()

    @tf.function()
    def compute_update_ratio(self, before_weights, after_weights, steps: int):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [
            after - before
            for after, before in zip(after_weights, before_weights)
        ]
        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [
            tf.math.reduce_euclidean_norm(w) for w in before_weights
        ]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.))
                  for d, w, tensor in zip(delta_norms, weight_norms,
                                          self.model.weights)
                  if not "moving" in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar("update_ratios/" + name, ratio, step=steps)
        # Filtering is hard, so just push infinities/NaNs to an unreasonably large value.
        ratios = [
            tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299,
                    lambda: 200.) for (_, r) in ratios
        ]
        tf.summary.histogram("update_ratios_log10",
                             tf.stack(ratios),
                             buckets=1000,
                             step=steps)

    def update_swa(self):
        num = self.swa_count.read_value()
        for (w, swa) in zip(self.model.weights, self.swa_weights):
            swa.assign(swa.read_value() * (num / (num + 1.)) + w.read_value() *
                       (1. / (num + 1.)))
        self.swa_count.assign(min(num + 1., self.swa_max_n))

    def save_swa_weights(self, filename: str):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        self.save_leelaz_weights(filename)
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def save_leelaz_weights(self, filename: str):
        numpy_weights = []
        for weight in self.model.weights:
            numpy_weights.append([weight.name, weight.numpy()])
        self.net.fill_net_v2(numpy_weights)
        self.net.save_proto(filename)

    @staticmethod
    def split_heads(inputs, batch_size: int, num_heads: int, depth: int):
        if num_heads < 2:
            return inputs
        reshaped = tf.reshape(inputs, (batch_size, 64, num_heads, depth))
        # (batch_size, num_heads, 64, depth)
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, name: str = None, inputs=None):

        # 0 h 64 d, 0 h d 64
        dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
        scaleDivisor = tf.pow(dk, 0.25)
        # q = q / scaleDivisor
        # k = k / scaleDivisor

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        batch_size = tf.shape(q)[0]
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        heads = scaled_attention_logits.shape[1]

        if self.use_smolgen:
            smolgen_weights = self.smolgen_weights(inputs, heads, self.smolgen_hidden_channels, self.smolgen_hidden_sz,
                                                   self.smolgen_gen_sz, name=name+"/smolgen", activation=self.smolgen_activation)
            scaled_attention_logits = scaled_attention_logits + smolgen_weights

        if self.use_logit_gating:
            scaled_attention_logits = Gating(name=name+"/gating")(scaled_attention_logits)

        # 0 h 64 64
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        # output shape = (b, h, 64, d)

        return output, scaled_attention_logits

    # multi-head attention in encoder blocks

    def mha(self, inputs, emb_size: int, d_model: int, num_heads: int, initializer, name: str, att_expansion: int=1):
        depth = d_model * att_expansion
        assert depth % num_heads == 0
        
        head_depth = depth // num_heads
        # query, key, and value vectors for self-attention
        # inputs b, 64, sz

        use_bias = not self.omit_qkv_biases

        q = tf.keras.layers.Dense(
            depth, name=name+"/wq", kernel_initializer="glorot_normal", use_bias=use_bias)(inputs)
        k = tf.keras.layers.Dense(
            depth, name=name+"/wk", kernel_initializer="glorot_normal", use_bias=use_bias)(inputs)
        v = tf.keras.layers.Dense(
            depth, name=name+"/wv", kernel_initializer=initializer, use_bias=use_bias)(inputs)

        # split q, k and v into smaller vectors of size "depth" -- one for each head in multi-head attention
        batch_size = tf.shape(q)[0]

        q = self.split_heads(q, batch_size, num_heads, head_depth)
        k = self.split_heads(k, batch_size, num_heads, head_depth)
        v = self.split_heads(v, batch_size, num_heads, head_depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, name=name, inputs=inputs)

        if num_heads > 1:
            scaled_attention = tf.transpose(scaled_attention,
                                            perm=[0, 2, 1, 3])
            scaled_attention = tf.reshape(
                scaled_attention,
                (batch_size, -1, depth))  # concatenate heads

        # final dense layer
        output = tf.keras.layers.Dense(
            emb_size, name=name + "/dense", kernel_initializer=initializer, use_bias=not self.omit_other_biases)(scaled_attention)
        return output, attention_weights

    def recurrent_kda(self, q, k, v, log_decay, beta, name=None):
        output_dtype = v.dtype
        outputs = KDARecurrence(
            chunk_size=self.kda_chunk_size,
            dtype=tf.float32, name=name)([q, k, v, log_decay, beta])
        return tf.cast(outputs, output_dtype)

    def kda(self, inputs, emb_size: int, num_heads: int,
            initializer, name: str):
        use_qkv_bias = not self.omit_qkv_biases
        use_other_bias = not self.omit_other_biases
        key_depth = num_heads * self.kda_key_dim
        value_depth = num_heads * self.kda_value_dim
        batch_size = tf.shape(inputs)[0]

        q = tf.keras.layers.Dense(
            key_depth, name=name + "/wq", kernel_initializer="glorot_normal",
            use_bias=use_qkv_bias)(inputs)
        k = tf.keras.layers.Dense(
            key_depth, name=name + "/wk", kernel_initializer="glorot_normal",
            use_bias=use_qkv_bias)(inputs)
        v = tf.keras.layers.Dense(
            value_depth, name=name + "/wv", kernel_initializer=initializer,
            use_bias=use_qkv_bias)(inputs)

        raw_decay = tf.keras.layers.Dense(
            self.kda_gate_rank, name=name + "/decay_a",
            kernel_initializer=initializer, use_bias=use_other_bias)(inputs)
        raw_decay = tf.keras.layers.Dense(
            key_depth, name=name + "/decay_b",
            kernel_initializer=initializer, use_bias=use_other_bias)(raw_decay)
        raw_decay = tf.reshape(
            raw_decay, [batch_size, 64, num_heads, self.kda_key_dim])
        log_decay = KDALogDecay(
            num_heads, self.kda_key_dim, name=name)(raw_decay)

        beta = tf.keras.layers.Dense(
            num_heads, name=name + "/beta", kernel_initializer=initializer,
            use_bias=use_other_bias)(inputs)
        beta = tf.sigmoid(beta)

        q = tf.reshape(q, [batch_size, 64, num_heads, self.kda_key_dim])
        k = tf.reshape(k, [batch_size, 64, num_heads, self.kda_key_dim])
        v = tf.reshape(v, [batch_size, 64, num_heads, self.kda_value_dim])

        heads_per_direction = num_heads // len(self.kda_directions)
        head_slices = [
            slice(index * heads_per_direction,
                  (index + 1) * heads_per_direction)
            for index in range(len(self.kda_directions))
        ]
        orders = [tf.constant(KDA_TRAVERSALS[direction], dtype=tf.int32)
                  for direction in self.kda_directions]

        # All directions share one recurrence call: each head group is permuted
        # into its own traversal order first, so the scan runs once, not once
        # per direction.
        ordered = self.recurrent_kda(
            tf.concat([tf.gather(q[:, :, group, :], order, axis=1)
                       for group, order in zip(head_slices, orders)], axis=2),
            tf.concat([tf.gather(k[:, :, group, :], order, axis=1)
                       for group, order in zip(head_slices, orders)], axis=2),
            tf.concat([tf.gather(v[:, :, group, :], order, axis=1)
                       for group, order in zip(head_slices, orders)], axis=2),
            tf.concat([tf.gather(log_decay[:, :, group, :], order, axis=1)
                       for group, order in zip(head_slices, orders)], axis=2),
            tf.concat([tf.gather(beta[:, :, group], order, axis=1)
                       for group, order in zip(head_slices, orders)], axis=2),
            name=name + "/scan")

        mixed = tf.concat(
            [tf.gather(ordered[:, :, group, :], tf.argsort(order), axis=1)
             for group, order in zip(head_slices, orders)], axis=2)
        mixed = tf.reshape(mixed, [batch_size, 64, value_depth])

        if self.kda_output_gate:
            gate = tf.keras.layers.Dense(
                self.kda_gate_rank, name=name + "/gate_a",
                kernel_initializer=initializer,
                use_bias=use_other_bias)(inputs)
            gate = tf.keras.layers.Dense(
                value_depth, name=name + "/gate_b",
                kernel_initializer=initializer,
                use_bias=use_other_bias)(gate)
            mixed = mixed * tf.cast(tf.sigmoid(gate), mixed.dtype)

        return tf.keras.layers.Dense(
            emb_size, name=name + "/dense", kernel_initializer=initializer,
            use_bias=use_other_bias)(mixed)

    # 2-layer dense feed-forward network in encoder blocks
    def ffn(self, inputs, emb_size: int, dff: int, initializer, name: str):
        if isinstance(self.ffn_activation, str):
            activation = tf.keras.activations.get(self.ffn_activation)
        else:
            activation = self.ffn_activation

        dense1 = tf.keras.layers.Dense(
            dff, name=name + "/dense1", kernel_initializer=initializer, activation=activation, use_bias=not self.omit_other_biases)(inputs)

        out = tf.keras.layers.Dense(
            emb_size, name=name + "/dense2", kernel_initializer=initializer, use_bias=not self.omit_other_biases)(dense1)
        return out

    def encoder_layer(self, inputs, emb_size: int, d_model: int, num_heads: int, dff: int, name: str, training: bool, mixer_type: str = "mha"):
        # DeepNorm
        alpha = tf.cast(tf.math.pow(
            2. * self.encoder_layers, 0.25), self.model_dtype)
        beta = tf.cast(tf.math.pow(
            8. * self.encoder_layers, -0.25), self.model_dtype)


        xavier_norm = tf.keras.initializers.VarianceScaling(
            scale=beta, mode="fan_avg", distribution="truncated_normal", seed=42)

        if mixer_type == "mha":
            attn_output, attn_wts = self.mha(
                inputs, emb_size, d_model, num_heads, xavier_norm,
                name=name + "/mha")
        elif mixer_type == "kda":
            attn_output = self.kda(
                inputs, emb_size, num_heads, xavier_norm,
                name=name + "/kda")
            attn_wts = None
        else:
            raise ValueError("Unknown encoder mixer: {}".format(mixer_type))

        # dropout for weight regularization
        attn_output = tf.keras.layers.Dropout(
            self.dropout_rate, name=name + "/dropout1")(attn_output, training=training)

        # skip connection + layernorm
        out1 = alpha * inputs + attn_output
        # out1 = inputs
        if not self.skip_first_ln:
            out1 = self.encoder_norm(
                name=name+"/ln1")(out1)

        # feed-forward network
        ffn_output = self.ffn(out1, emb_size, dff,
                              xavier_norm, name=name + "/ffn")
        ffn_output = tf.keras.layers.Dropout(
            self.dropout_rate, name=name + "/dropout2")(ffn_output, training=training)

        out2 = alpha * out1 + ffn_output

        out2 = self.encoder_norm(
            name=name+"/ln2")(out2)

        return out2, attn_wts

    def smolgen_weights(self, inputs, heads: int, hidden_channels: int, hidden_sz: int, gen_sz: int, name: str, activation="swish"):
        compressed = tf.keras.layers.Dense(
            hidden_channels, name=name+"/compress", use_bias=False)(inputs)
        compressed = tf.reshape(compressed, [-1, 64 * hidden_channels])
        hidden = tf.keras.layers.Dense(
            hidden_sz, name=name+"/hidden1_dense", activation=activation)(compressed)

        hidden = tf.keras.layers.LayerNormalization(
            name=name+"/hidden1_ln")(hidden)
        gen_from = tf.keras.layers.Dense(
            heads * gen_sz, name=name+"/gen_from", activation=activation)(hidden)
        gen_from = tf.keras.layers.LayerNormalization(
            name=name+"/gen_from_ln", center=True)(gen_from)
        gen_from = tf.reshape(gen_from, [-1, heads, gen_sz])

        out = self.smol_weight_gen_dense(gen_from)
        return tf.reshape(out, [-1, heads, 64, 64])

    def construct_net(self, inputs, name: str = ""):
        # Policy head
        assert self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION
        # TODO: re-add support for policy encoder blocks
        # do some input processing
        if self.use_smolgen:
            self.smol_weight_gen_dense = tf.keras.layers.Dense(
                64 * 64, name=name+"smol_weight_gen", use_bias=False)

        if self.embedding_style == "new":
            inputs = tf.cast(inputs, self.model_dtype)
            flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
            flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])

            pos_info = flow[..., :12]
            pos_info_flat = tf.reshape(pos_info, [-1, 64 * 12])

            pos_info_processed = tf.keras.layers.Dense(
                64*self.embedding_dense_sz, name=name+"embedding/preprocess")(pos_info_flat)
            pos_info = tf.reshape(pos_info_processed,
                                  [-1, 64, self.embedding_dense_sz])
            flow = tf.keras.layers.Concatenate()([flow, pos_info])

            # square embedding
            flow = tf.keras.layers.Dense(self.embedding_size, kernel_initializer="glorot_normal",
                                         kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                         name=name+"embedding")(flow)
            flow = self.encoder_norm(
                name=name+"embedding/ln")(flow)
            flow = ma_gating(flow, name=name+'embedding')

            # DeepNorm
            alpha = tf.cast(tf.math.pow(
                2. * self.encoder_layers, 0.25), self.model_dtype)
            beta = tf.cast(tf.math.pow(
                8. * self.encoder_layers, -0.25), self.model_dtype)



            xavier_norm = tf.keras.initializers.VarianceScaling(
                scale=beta, mode="fan_avg", distribution="truncated_normal", seed=42)

            # feed-forward network
            ffn_output = self.ffn(flow, self.embedding_size, self.encoder_dff,
                                  xavier_norm, name=name + "embedding/ffn")



            flow = self.encoder_norm(
                name=name+"embedding/ffn_ln")(alpha * flow + ffn_output)

        elif self.embedding_style == "old":
            flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
            flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])

            # square embedding
            flow = tf.keras.layers.Dense(self.embedding_size,
                                         kernel_initializer='glorot_normal',
                                         kernel_regularizer=self.l2reg,
                                         activation=self.DEFAULT_ACTIVATION,
                                         name='embedding')(flow)

            # !!! input gate
            flow = ma_gating(flow, name='embedding')

        else:
            raise ValueError(
                "Unknown embedding style: {}".format(self.embedding_style))

        attn_wts = []
        for i in range(self.encoder_layers):
            mixer_type = self.encoder_mixers[i]
            flow, attn_wts_l = self.encoder_layer(flow, self.embedding_size, self.encoder_d_model,
                                                  self.encoder_heads, self.encoder_dff,
                                                  name=name+"encoder_{}".format(i + 1), training=True,
                                                  mixer_type=mixer_type)
            if attn_wts_l is not None:
                attn_wts.append(attn_wts_l)

        flow_ = flow

        policy_tokens = tf.keras.layers.Dense(self.pol_embedding_size, kernel_initializer="glorot_normal",
                                              kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                              name=name+"policy/embedding")(flow_)

        def policy_head(name, activation=None, depth=None, opponent=False):
            if depth is None:
                depth = self.policy_d_model

            # reverse the tokens along the square (second) dimension to get the opponent's perspective
            tokens = tf.reverse(policy_tokens, axis=[
                1]) if opponent else policy_tokens

            # create queries and keys for policy self-attention
            queries = tf.keras.layers.Dense(depth, kernel_initializer="glorot_normal",
                                            name=name+"/attention/wq")(tokens)
            keys = tf.keras.layers.Dense(depth, kernel_initializer="glorot_normal",
                                         name=name+"/attention/wk")(tokens)

            # POLICY SELF-ATTENTION: self-attention weights are interpreted as from->to policy
            # Bx64x64 (from 64 queries, 64 keys)
            matmul_qk = tf.matmul(queries, keys, transpose_b=True)
            # queries = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer="glorot_normal",
            #                                 name="policy/attention/wq")(flow)
            # keys = tf.keras.layers.Dense(self.policy_d_model, kernel_initializer="glorot_normal",
            #                              name="policy/attention/wk")(flow)

            # PAWN PROMOTION: create promotion logits using scalar offsets generated from the promotion-rank keys
            # constant for scaling
            dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))
            promotion_keys = keys[:, -8:, :]
            # queen, rook, bishop, knight order
            promotion_offsets = tf.keras.layers.Dense(4, kernel_initializer="glorot_normal",
                                                      name=name+"/attention/ppo", use_bias=False)(promotion_keys)
            promotion_offsets = tf.transpose(
                promotion_offsets, perm=[0, 2, 1]) * dk  # Bx4x8
            # knight offset is added to the other three
            promotion_offsets = promotion_offsets[:,
                                                  :3, :] + promotion_offsets[:, 3:4, :]

            # q, r, and b promotions are offset from the default promotion logit (knight)
            # default traversals from penultimate rank to promotion rank
            n_promo_logits = matmul_qk[:, -16:-8, -8:]
            q_promo_logits = tf.expand_dims(
                n_promo_logits + promotion_offsets[:, 0:1, :], axis=3)  # Bx8x8x1
            r_promo_logits = tf.expand_dims(
                n_promo_logits + promotion_offsets[:, 1:2, :], axis=3)
            b_promo_logits = tf.expand_dims(
                n_promo_logits + promotion_offsets[:, 2:3, :], axis=3)
            promotion_logits = tf.concat(
                [q_promo_logits, r_promo_logits, b_promo_logits], axis=3)  # Bx8x8x3
            # logits now alternate a7a8q,a7a8r,a7a8b,...,
            promotion_logits = tf.reshape(promotion_logits, [-1, 8, 24])

            # scale the logits by dividing them by sqrt(d_model) to stabilize gradients
            # Bx8x24 (8 from-squares, 3x8 promotions)
            promotion_logits = promotion_logits / dk
            # Bx64x64 (64 from-squares, 64 to-squares)
            policy_attn_logits = matmul_qk / dk

            attn_wts.append(promotion_logits)
            attn_wts.append(policy_attn_logits)

            # APPLY POLICY MAP: output becomes Bx1856
            h_fc1 = ApplyAttentionPolicyMap(
                name=name+"/attention_map")(policy_attn_logits, promotion_logits)

            if activation is not None:
                h_fc1 = tf.keras.layers.Activation(activation)(h_fc1)

            # Value head
            assert self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION and self.encoder_layers > 0

            return h_fc1

        def future_head(name):
            # hack so checkpointing works
            return tf.keras.layers.Dense(2, name=name+"/attention/wq")(policy_tokens)


        aux_depth = self.cfg['model'].get('policy_d_aux', self.policy_d_model)

        policy = policy_head(name="policy/vanilla")

        policy_optimistic_st = policy_head(
            name="policy/optimistic_st") if self.cfg['model'].get('policy_optimistic_st', False) else None

        policy_soft = policy_head(
            name="policy/soft", depth=aux_depth) if self.cfg['model'].get('soft_policy', False) else None
        policy_opponent = future_head(name="policy/opponent") if self.cfg['model'].get(
            'policy_opponent', False) else None
        policy_next = future_head(name="policy/next") if self.cfg['model'].get(
            'policy_next', False) else None

        def value_head(name, wdl=True, use_err=True, use_cat=True):
            embedded_val = tf.keras.layers.Dense(self.val_embedding_size, kernel_initializer="glorot_normal",
                                                 kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                                 name=name+"/embedding")(flow)
            h_val_flat = tf.keras.layers.Flatten()(embedded_val)
            h_fc2 = tf.keras.layers.Dense(128,
                                          kernel_initializer="glorot_normal",
                                          kernel_regularizer=self.l2reg,
                                          activation=self.DEFAULT_ACTIVATION,
                                          name=name+"/dense1")(h_val_flat)
            # WDL head
            if wdl:
                value = tf.keras.layers.Dense(3,
                                              kernel_initializer="glorot_normal",
                                              kernel_regularizer=self.l2reg,
                                              bias_regularizer=self.l2reg,
                                              name=name+"/dense2")(h_fc2)
            else:
                value = tf.keras.layers.Dense(1,
                                              kernel_initializer="glorot_normal",
                                              kernel_regularizer=self.l2reg,
                                              activation="tanh",
                                              name=name+"/dense2")(h_fc2)

            if use_err:
                # Shouldn't be more than 1
                value_err = tf.keras.layers.Dense(
                    1, kernel_initializer="glorot_normal", name=name+"/dense_error", activation="sigmoid")(h_fc2)
            else:
                value_err = None

            if use_cat:
                value_cat = tf.keras.layers.Dense(
                    self.categorical_value_buckets, kernel_initializer="glorot_normal", name=name+"/dense_cat")(h_fc2)
            else:
                value_cat = None

            return value, value_err, value_cat

        value_winner, value_winner_err, value_winner_cat = value_head(
            name="value/winner", wdl=self.wdl, use_err=False, use_cat=False)
        value_q, value_q_err, value_q_cat = value_head(
            name="value/q", wdl=self.wdl, use_err=True) if self.cfg['model'].get('value_q', False) else (None, None, None)
        value_st, value_st_err, value_st_cat = value_head(
            name="value/st", wdl=self.wdl, use_err=True) if self.cfg['model'].get('value_st', False) else (None, None, None)

        # Moves left head
        if self.moves_left:
            embedded_mov = tf.keras.layers.Dense(self.mov_embedding_size, kernel_initializer="glorot_normal",
                                                 kernel_regularizer=self.l2reg, activation=self.DEFAULT_ACTIVATION,
                                                 name=name+"moves_left/embedding")(flow)
            h_mov_flat = tf.keras.layers.Flatten()(embedded_mov)

            h_fc4 = tf.keras.layers.Dense(
                128,
                kernel_initializer="glorot_normal",
                kernel_regularizer=self.l2reg,
                activation=self.DEFAULT_ACTIVATION,
                name=name+"moves_left/dense1")(h_mov_flat)

            moves_left = tf.keras.layers.Dense(1,
                                               kernel_initializer="glorot_normal",
                                               kernel_regularizer=self.l2reg,
                                               activation="relu",
                                               name=name+"moves_left/dense2")(h_fc4)
        else:
            moves_left = None

        # attention weights added as optional output for analysis -- ignored by backend
        outputs = {
            "policy": policy,
            "policy_optimistic_st": policy_optimistic_st,
            "policy_soft": policy_soft,
            "policy_opponent": policy_opponent,
            "policy_next": policy_next,
            "value_winner": value_winner,
            "value_q": value_q,
            "value_q_err": value_q_err,
            "value_q_cat": value_q_cat,
            "value_st": value_st,
            "value_st_err": value_st_err,
            "value_st_cat": value_st_cat,
            "moves_left": moves_left,
        }

        if self.return_attn_wts and attn_wts:
            outputs["attn_wts"] = attn_wts

        # Tensorflow does not accept None values in the output dictionary
        none_keys = []
        for key in outputs:
            if outputs[key] is None:
                none_keys.append(key)

        for key in none_keys:
            del outputs[key]

        for key in outputs:
            try:
                outputs[key] = tf.cast(outputs[key], tf.float32)
            except:
                assert key == "attn_wts"
                # don't want to cast since the memory will jump
                # out = []
                # for t in outputs[key]:
                #     out.append(tf.cast(t, tf.float32))
                # outputs[key] = out

        return outputs

    def set_sparsity_patterns(self):
        sparsity_patterns = {}
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and "encoder" in layer.name and "smolgen" not in layer.name:
                kernel = layer.kernel
                # 2 out of 4 sparsity pattern
                in_channels = kernel.shape[0]
                out_channels = kernel.shape[1]

                kernel_abs = tf.abs(kernel)
                kernel_abs = tf.reshape(kernel_abs, [-1, 4])

                top_2 = tf.math.top_k(kernel_abs, k=2, sorted=True)
                second_largest = top_2.values[:, 1:2]
                comparison = tf.math.greater_equal(kernel_abs, second_largest)
                comparison = tf.cast(comparison, tf.float32)
                comparison = tf.reshape(
                    comparison, [in_channels, out_channels])

                sparsity_patterns[layer.name] = comparison

        self.sparsity_patterns = sparsity_patterns

    def apply_sparsity(self):
        assert hasattr(self, "sparsity_patterns"), "Sparsity patterns not set"
        for layer in self.model.layers:
            if layer.name in self.sparsity_patterns:
                kernel = layer.kernel
                kernel.assign(kernel * self.sparsity_patterns[layer.name])