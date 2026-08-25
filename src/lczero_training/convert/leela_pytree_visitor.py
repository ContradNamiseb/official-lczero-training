import math
from typing import Any, Optional

from flax import nnx

from proto import net_pb2


class LeelaPytreeWeightsVisitor:
    def __init__(self, nnx_state: nnx.State, leela_net: net_pb2.Net) -> None:
        self.leela_net = leela_net
        self.nnx_state = nnx_state

    def run(self) -> None:
        state = self.nnx_state
        weights = self.leela_net.weights
        self.embedding_block(state["embedding"], weights)
        self.encoder_tower(state["encoders"], weights)
        self.policy_heads(state, weights.policy_heads)
        for head_name in ["winner", "q", "st"]:
            if head_name in state["value_heads"]:
                self.value_head(
                    state["value_heads"][head_name],
                    getattr(weights.value_heads, head_name),
                )
        for head_name in ["main"]:
            assert head_name in state["movesleft_heads"], (
                f"movesleft head {head_name} missing in state"
            )
            self.movesleft_head(state["movesleft_heads"][head_name], weights)

    def embedding_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        self.matmul(
            nnx_dict["preprocess"],
            weights.ip_emb_preproc_w,
            weights.ip_emb_preproc_b,
        )
        self.matmul(
            nnx_dict["embedding"],
            weights.ip_emb_w,
            weights.ip_emb_b,
        )
        self.layernorm(
            nnx_dict["norm"],
            weights.ip_emb_ln_gammas,
            weights.ip_emb_ln_betas,
        )
        self.tensor(
            nnx_dict["ma_gating"]["mult_gate"]["gate"], weights.ip_mult_gate
        )
        self.tensor(
            nnx_dict["ma_gating"]["add_gate"]["gate"], weights.ip_add_gate
        )
        self.ffn(nnx_dict["ffn"], weights.ip_emb_ffn)
        self.layernorm(
            nnx_dict["out_norm"],
            weights.ip_emb_ffn_ln_gammas,
            weights.ip_emb_ffn_ln_betas,
        )

    def encoder_tower(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        # Shared layer is stored at the point of the first usage. A tower
        # may have KDA blocks (no smolgen) before its first MHA block, so
        # this has to scan for the first MHA block rather than assume
        # layers[0] -- a KDA-only tower has no smolgen dense at all.
        layers = nnx_dict["encoders"]["layers"]
        # Iterating an nnx.State directly yields its (integer) keys, not
        # the nested per-layer states -- index explicitly, matching the
        # loop just below.
        first_mha = next(
            (layers[i] for i in range(len(layers)) if "mha" in layers[i]),
            None,
        )
        if first_mha is not None:
            self.matmul(
                first_mha["mha"]["smolgen"]["weight_gen_dense"],
                weights.smolgen_w,
                None,
            )

        # assert len(nnx_dict["encoders"]["layers"]) == len(weights.encoder)
        for i in range(len(layers)):
            self.encoder_block(layers[i], weights.encoder[i])

    def encoder_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.EncoderLayer
    ) -> None:
        if "mha" in nnx_dict:
            self.mha(nnx_dict["mha"], weights.mha)
        elif "mixer" in nnx_dict:
            self.kda_mixer(nnx_dict["mixer"], weights.kda)
        else:
            raise ValueError(
                "Encoder block has neither 'mha' nor 'mixer' in state."
            )
        self.layernorm(nnx_dict["ln1"], weights.ln1_gammas, weights.ln1_betas)
        self.ffn(nnx_dict["ffn"], weights.ffn)
        self.layernorm(nnx_dict["ln2"], weights.ln2_gammas, weights.ln2_betas)

    def mha(self, nnx_dict: nnx.State, weights: net_pb2.Weights.MHA) -> None:
        self.matmul(nnx_dict["q"], weights.q_w, weights.q_b)
        self.matmul(nnx_dict["k"], weights.k_w, weights.k_b)
        self.matmul(nnx_dict["v"], weights.v_w, weights.v_b)
        self.smolgen(nnx_dict["smolgen"], weights.smolgen)
        self.matmul(nnx_dict["output_dense"], weights.dense_w, weights.dense_b)

    def kda_mixer(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.KDA
    ) -> None:
        """Structural (shape-only) mapping between a KdaMixer's NNX state
        and Weights.KDA. Scalar config fields (key_dim, value_dim,
        gate_rank, output_gate, output_rms_norm, local_conv) are not part
        of the NNX pytree -- they come from ModelConfig.encoder.kda, not
        from here, so callers that need to write or read them handle that
        separately (see JaxToLeela.encoder_block).
        """
        if "local_conv" in nnx_dict:
            self.kda_local_conv(
                nnx_dict["local_conv"]["conv"],
                weights.local_conv_w,
                weights.local_conv_b,
            )
        self.matmul(nnx_dict["q"], weights.q_w, weights.q_b)
        self.matmul(nnx_dict["k"], weights.k_w, weights.k_b)
        self.matmul(nnx_dict["v"], weights.v_w, weights.v_b)
        self.matmul(nnx_dict["decay_a"], weights.decay_a_w, weights.decay_a_b)
        self.matmul(nnx_dict["decay_b"], weights.decay_b_w, weights.decay_b_b)
        self.matmul(nnx_dict["beta"], weights.beta_w, weights.beta_b)
        self.tensor(nnx_dict["log_decay"]["a_log"], weights.a_log)
        self.tensor(nnx_dict["log_decay"]["dt_bias"], weights.dt_bias)
        if "gate_a" in nnx_dict:
            self.matmul(nnx_dict["gate_a"], weights.gate_a_w, weights.gate_a_b)
            self.matmul(nnx_dict["gate_b"], weights.gate_b_w, weights.gate_b_b)
        if "rms_norm_gammas" in nnx_dict:
            self.tensor(nnx_dict["rms_norm_gammas"], weights.out_norm_gammas)
        self.matmul(nnx_dict["output_dense"], weights.dense_w, weights.dense_b)

    def kda_local_conv(
        self,
        nnx_dict: nnx.State,
        kernel_weights: net_pb2.Weights.Layer,
        bias_weights: net_pb2.Weights.Layer,
    ) -> None:
        """Depthwise conv kernel layout differs between Flax and the engine
        (Flax: (kh, kw, 1, channels); engine: flat c*9 + kr*3 + kf), so this
        cannot go through the generic matmul()/tensor() helpers -- each
        direction (JaxToLeela / LeelaToJax) must override this to apply its
        own layout conversion.
        """
        raise NotImplementedError(
            "kda_local_conv must be overridden by a direction-specific "
            "visitor; the depthwise kernel layout conversion is not "
            "symmetric and cannot have a generic default."
        )

    def smolgen(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.Smolgen
    ) -> None:
        self.matmul(nnx_dict["compress"], weights.compress, None)
        self.matmul(nnx_dict["dense1"], weights.dense1_w, weights.dense1_b)
        self.layernorm(nnx_dict["ln1"], weights.ln1_gammas, weights.ln1_betas)
        self.matmul(nnx_dict["dense2"], weights.dense2_w, weights.dense2_b)
        self.layernorm(nnx_dict["ln2"], weights.ln2_gammas, weights.ln2_betas)

    def layernorm(
        self,
        nnx_dict: nnx.State,
        scales: net_pb2.Weights.Layer,
        biases: net_pb2.Weights.Layer,
    ) -> None:
        self.tensor(nnx_dict["scale"], scales)
        self.tensor(nnx_dict["bias"], biases)

    def policy_heads(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.PolicyHeads
    ) -> None:
        if "policy_embedding_shared" in nnx_dict:
            self.matmul(
                nnx_dict["policy_embedding_shared"],
                weights.ip_pol_w,
                weights.ip_pol_b,
            )
        policy_heads_dict = nnx_dict["policy_heads"]
        for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
            if head_name in policy_heads_dict:
                self.policy_head(
                    policy_heads_dict[head_name], getattr(weights, head_name)
                )

    def policy_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.PolicyHead
    ) -> None:
        if "tokens" in nnx_dict:
            self.matmul(nnx_dict["tokens"], weights.ip_pol_w, weights.ip_pol_b)
        self.matmul(nnx_dict["q"], weights.ip2_pol_w, weights.ip2_pol_b)
        self.matmul(nnx_dict["k"], weights.ip3_pol_w, weights.ip3_pol_b)
        self.matmul(nnx_dict["promotion_dense"], weights.ip4_pol_w, None)

    def value_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.ValueHead
    ) -> None:
        self.matmul(nnx_dict["embed"], weights.ip_val_w, weights.ip_val_b)
        self.matmul(nnx_dict["dense1"], weights.ip1_val_w, weights.ip1_val_b)
        self.matmul(nnx_dict["wdl"], weights.ip2_val_w, weights.ip2_val_b)
        if "error" in nnx_dict:
            self.matmul(
                nnx_dict["error"], weights.ip_val_err_w, weights.ip_val_err_b
            )
        if "categorical" in nnx_dict:
            self.matmul(
                nnx_dict["categorical"],
                weights.ip_val_cat_w,
                weights.ip_val_cat_b,
            )

    def movesleft_head(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        self.matmul(nnx_dict["embed"], weights.ip_mov_w, weights.ip_mov_b)
        self.matmul(nnx_dict["dense1"], weights.ip1_mov_w, weights.ip1_mov_b)
        self.matmul(nnx_dict["out"], weights.ip2_mov_w, weights.ip2_mov_b)

    def ffn(self, nnx_dict: nnx.State, ffn: net_pb2.Weights.FFN) -> None:
        self.matmul(nnx_dict["linear1"], ffn.dense1_w, ffn.dense1_b)
        self.matmul(nnx_dict["linear2"], ffn.dense2_w, ffn.dense2_b)

    def matmul(
        self,
        nnx_dict: nnx.State,
        weights: net_pb2.Weights.Layer,
        biases: Optional[net_pb2.Weights.Layer],
    ) -> None:
        self.tensor(nnx_dict["kernel"], weights)
        if biases and "bias" in nnx_dict:
            self.tensor(nnx_dict["bias"], biases)
        elif biases:
            self.zero_bias(nnx_dict["kernel"], biases)
        else:
            assert "bias" not in nnx_dict

    def zero_bias(
        self,
        kernel_param: Any,
        biases: net_pb2.Weights.Layer,
    ) -> None:
        pass

    def tensor(
        self,
        param: Any,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        print(
            param.shape,
            len(leela.params) // 2,
            math.prod(param.shape),
        )
        assert len(leela.params) // 2 == math.prod(param.shape)
        assert len(leela.params) != 0
