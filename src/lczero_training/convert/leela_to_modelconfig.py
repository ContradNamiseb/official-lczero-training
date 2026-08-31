from proto import hlo_pb2, model_config_pb2, net_pb2


def _defaultactivation_to_activation(
    activation: net_pb2.NetworkFormat.DefaultActivation,
) -> net_pb2.NetworkFormat.ActivationFunction:
    return {
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU: net_pb2.NetworkFormat.ACTIVATION_RELU,
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH: net_pb2.NetworkFormat.ACTIVATION_MISH,
    }[activation]


# Inverse of _KDA_DIRECTION_TO_ENUM in convert/jax_to_leela.py. Kept as an
# independent mapping (rather than importing that module) to avoid a
# leela_to_jax.py -> leela_to_modelconfig.py -> jax_to_leela.py import cycle,
# since leela_to_jax.py already imports from both.
_KDA_ENUM_TO_DIRECTION = {
    net_pb2.NetworkFormat.KDA_DIRECTION_RANK_FORWARD: "rank_forward",
    net_pb2.NetworkFormat.KDA_DIRECTION_RANK_REVERSE: "rank_reverse",
    net_pb2.NetworkFormat.KDA_DIRECTION_FILE_FORWARD: "file_forward",
    net_pb2.NetworkFormat.KDA_DIRECTION_FILE_REVERSE: "file_reverse",
    net_pb2.NetworkFormat.KDA_DIRECTION_DIAG_FORWARD: "diag_forward",
    net_pb2.NetworkFormat.KDA_DIRECTION_DIAG_REVERSE: "diag_reverse",
    net_pb2.NetworkFormat.KDA_DIRECTION_ANTI_DIAG_FORWARD: "anti_diag_forward",
    net_pb2.NetworkFormat.KDA_DIRECTION_ANTI_DIAG_REVERSE: "anti_diag_reverse",
    # Imported as explicit direction names, leaving KdaConfig.serpentine
    # false: the substitution is idempotent, so a config naming the
    # serpentine walks directly builds exactly the same model.
    net_pb2.NetworkFormat.KDA_DIRECTION_RANK_SERPENTINE: "rank_serpentine",
    net_pb2.NetworkFormat.KDA_DIRECTION_RANK_SERPENTINE_REVERSE: (
        "rank_serpentine_reverse"
    ),
    net_pb2.NetworkFormat.KDA_DIRECTION_FILE_SERPENTINE: "file_serpentine",
    net_pb2.NetworkFormat.KDA_DIRECTION_FILE_SERPENTINE_REVERSE: (
        "file_serpentine_reverse"
    ),
}


def leela_to_modelconfig(
    leela_net: net_pb2.Net,
    weights_dtype: hlo_pb2.XlaShapeProto.Type,
    compute_dtype: hlo_pb2.XlaShapeProto.Type,
) -> model_config_pb2.ModelConfig:
    assert weights_dtype == hlo_pb2.XlaShapeProto.F32, (
        "Only float32 weights are supported."
    )
    assert leela_net.format.weights_encoding == net_pb2.Format.LINEAR16
    leela_net_format = leela_net.format.network_format
    model_config = model_config_pb2.ModelConfig()

    model_config.defaults.compute_dtype = compute_dtype
    model_config.defaults.activation = _defaultactivation_to_activation(
        leela_net_format.default_activation
    )
    model_config.defaults.ffn_activation = (
        leela_net_format.ffn_activation or model_config.defaults.activation
    )
    assert (
        leela_net_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    ), "Only dense positional embedding is supported, got {}".format(
        net_pb2.NetworkFormat.InputEmbeddingFormat.Name(
            leela_net_format.input_embedding
        )
    )
    assert leela_net_format.policy == net_pb2.NetworkFormat.POLICY_ATTENTION, (
        "Only attention policy is supported, got {}".format(
            net_pb2.NetworkFormat.PolicyFormat.Name(leela_net_format.policy)
        )
    )
    assert leela_net_format.value == net_pb2.NetworkFormat.VALUE_WDL, (
        "Only WDL value is supported, got {}".format(
            net_pb2.NetworkFormat.ValueFormat.Name(leela_net_format.value)
        )
    )
    assert leela_net_format.moves_left == net_pb2.NetworkFormat.MOVES_LEFT_V1, (
        "Only V1 moves left format is supported, got {}".format(
            net_pb2.NetworkFormat.MovesLeftFormat.Name(
                leela_net_format.moves_left
            )
        )
    )

    def size(x: net_pb2.Weights.Layer) -> int:
        return len(x.params) // 2

    assert leela_net_format.network in (
        net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT,
        net_pb2.NetworkFormat.NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT,
    )
    weights = leela_net.weights
    model_config.embedding.dense_size = size(weights.ip_emb_preproc_b) // 64
    model_config.embedding.embedding_size = size(weights.ip_emb_b)
    assert size(weights.ip_mult_gate) > 0
    assert size(weights.ip_add_gate) > 0
    model_config.embedding.dff = size(weights.ip_emb_ffn.dense1_b)

    model_config.encoder.num_blocks = len(weights.encoder)
    assert model_config.encoder.num_blocks > 0
    encoder = weights.encoder[0]
    model_config.encoder.heads = weights.headcount
    # dff/FFN width is shared structure, present on every block regardless
    # of its mixer, so layer 0 is representative for it even in a mixed
    # tower.
    model_config.encoder.dff = size(encoder.ffn.dense1_b)

    # Mixer type IS per-block (a real checkpoint can be e.g. [kda, kda,
    # kda, mha] -- see model/utils.py's encoder_mixer_pattern), so this
    # has to read every layer, not just layer 0.
    mixer_types = [layer.mixer for layer in weights.encoder]
    kda_enum = net_pb2.Weights.EncoderLayer.MIXER_KDA
    mha_enum = net_pb2.Weights.EncoderLayer.MIXER_MHA
    if all(m == mixer_types[0] for m in mixer_types):
        model_config.encoder.mixer_type = (
            model_config_pb2.MIXER_KDA
            if mixer_types[0] == kda_enum
            else model_config_pb2.MIXER_MHA
        )
    else:
        model_config.encoder.mixer_pattern.extend(
            model_config_pb2.MIXER_KDA
            if m == kda_enum
            else model_config_pb2.MIXER_MHA
            for m in mixer_types
        )

    kda_layer = next(
        (layer for layer in weights.encoder if layer.mixer == kda_enum), None
    )
    if kda_layer is not None:
        kda = kda_layer.kda
        model_config.encoder.kda.key_dim = kda.key_dim
        model_config.encoder.kda.value_dim = kda.value_dim
        model_config.encoder.kda.gate_rank = kda.gate_rank
        model_config.encoder.kda.output_gate = kda.output_gate
        model_config.encoder.kda.output_rms_norm = kda.output_rms_norm
        model_config.encoder.kda.local_conv = kda.local_conv
        model_config.encoder.kda.qkv_silu = kda.qkv_silu
        # chunk_size has no engine-side field -- it only controls how the
        # chunkwise-parallel algorithm splits work, not the model's math
        # (see kda_recurrence's docstring), so there is nothing to recover
        # here; leave it at the KdaConfig proto default.
        model_config.encoder.kda.directions.extend(
            _KDA_ENUM_TO_DIRECTION[d] for d in leela_net_format.kda_directions
        )

    mha_layer = next(
        (layer for layer in weights.encoder if layer.mixer == mha_enum), None
    )
    if mha_layer is not None:
        model_config.encoder.d_model = size(mha_layer.mha.q_b)

    if weights.HasField("smolgen_w"):
        assert mha_layer is not None, (
            "smolgen_w is set but no encoder layer uses MHA"
        )
        model_config.encoder.smolgen.activation = (
            leela_net_format.smolgen_activation
            or model_config.defaults.activation
        )
        model_config.encoder.smolgen.hidden_channels = (
            size(mha_layer.mha.smolgen.compress)
            // model_config.embedding.embedding_size
        )
        model_config.encoder.smolgen.gen_size = (
            size(mha_layer.mha.smolgen.dense2_b) // weights.headcount
        )
        model_config.encoder.smolgen.hidden_size = size(
            mha_layer.mha.smolgen.dense1_b
        )

    if weights.policy_heads.HasField("ip_pol_w"):
        model_config.shared_policy_embedding_size = size(
            weights.policy_heads.ip_pol_b
        )

    for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
        if weights.policy_heads.HasField(head_name):
            head = getattr(weights.policy_heads, head_name)
            assert size(head.ip2_pol_b) > 0
            # Mirrors PolicyHead.__init__'s own invariant: a head has its
            # own ip_pol_w exactly when the tower has no shared embedding.
            assert head.HasField("ip_pol_w") != model_config.HasField(
                "shared_policy_embedding_size"
            )
            policy_head = model_config.policy_head.add()
            policy_head.name = head_name
            if not model_config.HasField("shared_policy_embedding_size"):
                policy_head.embedding_size = size(head.ip_pol_b)
            policy_head.d_model = size(head.ip2_pol_b)

    for head_name in ["winner", "q", "st"]:
        if weights.value_heads.HasField(head_name):
            head = getattr(weights.value_heads, head_name)
            assert size(head.ip_val_b) > 0
            value_head = model_config.value_head.add()
            value_head.name = head_name
            value_head.num_channels = size(head.ip_val_b)
            if head.HasField("ip_val_err_w"):
                value_head.has_error_output = True
            if head.HasField("ip_val_cat_b"):
                value_head.num_categorical_buckets = size(head.ip_val_cat_b)

    movesleft_head = model_config.movesleft_head.add()
    movesleft_head.name = "main"
    movesleft_head.num_channels = size(weights.ip_mov_b)

    return model_config
