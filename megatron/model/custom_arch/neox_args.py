import torch

"""
Example usage:

    neox_args = NeoxArgs(
        init_method="normal",
        init_method_std=0.02,
        output_layer_init_method="normal",
        pos_emb='learned',
        norm='layernorm',
        hidden_size=embedding_len,
        hidden_dropout=0.05,
        bias_dropout_fusion=False,
        gpt_j_residual=False,
        gpt_j_tied=False,
        mlp_type='regular',
        precision=0,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=False,
        num_attention_heads=4,
        use_qk_layernorm=False,
        use_mup=False,
        mup_init_scale=1,
        model_parallel_size=1,
        rotary_pct=1,
        rotary_emb_base=10000,
        seq_length=None,
        params_dtype=torch.float,
        rope_fusion=False,
        attention_config=expand_attention_types([[["global"], 4]], 4),
        scaled_upper_triang_masked_softmax_fusion=False,
        scaled_masked_softmax_fusion=False,
        attention_dropout=0,
        use_cpu_initialization=False,
        activation='relu',
        bias_gelu_fusion=False,
        layernorm_epsilon='1e-05',
        layernorm_fusion=False,
        lazy_mpu_init=False,
        rank=1,
        local_rank=1,
        pipe_parallel_size=1,
        num_layers=4,
    )
"""


# ----------- NeoX CUSTOM ARGS----------- #
class NeoxArgs:
    def __init__(
        self,
        init_method="normal",
        init_method_std=0.02,
        output_layer_init_method="normal",
        opt_pos_emb_offset=None,
        mup_embedding_mult=None,
        mup_rp_embedding_mult=1,
        use_bnb_optimizer=False,
        pos_emb="learned",
        norm="layernorm",
        layernorm_epsilon=1e-05,
        layernorm_fusion=False,
        hidden_dropout=0.05,
        bias_dropout_fusion=False,
        gpt_j_residual=False,
        gpt_j_tied=False,
        mlp_type="regular",
        precision=0,
        hidden_size=64,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=False,
        num_attention_heads=4,
        use_qk_layernorm=False,
        use_mup=False,
        mup_init_scale=0.5,
        model_parallel_size=1,
        rotary_pct=1,
        rotary_emb_base=10000,
        seq_length=None,
        params_dtype=torch.float,
        rope_fusion=False,
        attention_config=None,
        scaled_upper_triang_masked_softmax_fusion=False,
        scaled_masked_softmax_fusion=False,
        attention_dropout=0,
        # have left out args for configure_sparse_attention
        use_cpu_initialization=False,
        activation='relu',
        bias_gelu_fusion=False,
        lazy_mpu_init=False,
        rank=1,
        local_rank=1,
        pipe_parallel_size=1,
        num_layers=4,
        distributed_backend="nccl",
        world_size=1,
        fp32_allreduce=False,
        checkpoint_num_layers=1,
        partition_activations=False,
        contiguous_checkpointing=False,
        checkpoint_in_cpu=False,
        synchronize_each_layer=False,
        profile_backward=False,
        seed=1234,
        tensorboard_writer=False,
        use_bias_in_attn_linear=False,
    ):

        self.init_method = init_method
        self.init_method_std = init_method_std
        self.output_layer_init_method = output_layer_init_method
        self.pos_emb = pos_emb
        self.norm = norm
        self.opt_pos_emb_offset=opt_pos_emb_offset
        self.mup_embedding_mult=mup_embedding_mult
        self.mup_rp_embedding_mult=mup_rp_embedding_mult
        self.use_bnb_optimizer=use_bnb_optimizer
        self.layernorm_epsilon = layernorm_epsilon
        self.layernorm_fusion = layernorm_fusion
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.gpt_j_residual = gpt_j_residual
        self.gpt_j_tied = gpt_j_tied
        self.mlp_type = mlp_type
        self.precision = precision
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.num_attention_heads = num_attention_heads
        self.use_qk_layernorm = use_qk_layernorm
        self.use_mup = use_mup
        self.mup_init_scale = mup_init_scale
        self.model_parallel_size = model_parallel_size
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.seq_length = seq_length
        self.params_dtype = params_dtype
        self.rope_fusion = rope_fusion
        self.attention_config = attention_config
        self.scaled_upper_triang_masked_softmax_fusion = (
            scaled_upper_triang_masked_softmax_fusion
        )
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.attention_dropout = attention_dropout
        self.use_cpu_initialization = use_cpu_initialization
        self.activation = activation
        self.bias_gelu_fusion = bias_gelu_fusion
        self.bias_dropout_fusion = bias_dropout_fusion
        self.lazy_mpu_init = lazy_mpu_init
        self.rank = rank
        self.local_rank = local_rank
        self.pipe_parallel_size = pipe_parallel_size
        self.distributed_backend = distributed_backend
        self.world_size = world_size
        self.fp32_allreduce = fp32_allreduce
        self.num_layers = num_layers
        self.checkpoint_num_layers = checkpoint_num_layers
        self.partition_activations = partition_activations
        self.contiguous_checkpointing = contiguous_checkpointing
        self.checkpoint_in_cpu = checkpoint_in_cpu
        self.synchronize_each_layer = synchronize_each_layer
        self.profile_backward = profile_backward
        self.seed = seed
        self.tensorboard_writer = tensorboard_writer
        self.use_bias_in_attn_linear = use_bias_in_attn_linear