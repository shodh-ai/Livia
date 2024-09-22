from megatron.model.transformer import ParallelTransformerLayer

from .multi_head_attention import MultiHeadAttention


# ------------ PARALLEL DECODER BLOCK ------------ #
class DecoderBlock(ParallelTransformerLayer):
    """
    Decoder Block

    Args:
        neox_args (dict): A dictionary containing NeoX-specific arguments.
        attention_mask_func (function): Function to compute attention mask.
        init_method (callable): Initialization method for the layer parameters.
        output_layer_init_method (callable): Initialization method for output layer.
        layer_number (int): The layer number.
        rpe (Optional[object]): Relative positional embeddings.
        rotary (bool): Whether to use rotary embeddings.
        use_cache (bool): Whether to use cache during inference.

    hidden_size(neox_arg) is our embedding_len

    Example:
        layer = DecoderBlock
                (
                    neox_args,
                    attention_mask_func,
                    init_methodk,
                    output_layer_init_method,
                    layer_number,
                )
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):

        # Pass the required arguments to the parent class __init__ method
        super().__init__(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
        )
        
        # Using our MultiHeadAttention
        self.attention = MultiHeadAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number
        )
        
        
class ParallelTransformerLayerPipe(DecoderBlock):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        output, moe_loss = super().forward(hidden_states, attention_mask)
        # auxiliary output
        self.last_moe_loss = moe_loss
        return output, attention_mask