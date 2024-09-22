import torch
import sys
from torch import nn

sys.path.append("/nlsasfs/home/shodhlab/ptarun/InferQ/JayDeep/gpt-neox-custom")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from megatron.model.transformer import ParallelTransformerLayer
from megatron.model.init_functions import get_init_methods
from megatron.model.gpt2_model import gpt2_attention_mask_func
from TransformerModel import MultiHeadAttention
from neox_args import NeoxArgs
from megatron.utils import expand_attention_types

from megatron.initialize import initialize_megatron
from multi_head_attention import MultiHeadAttention


# ------------ DECODER BLOCK ------------ #
class DecoderBlockOrig(nn.Module):
    def __init__(
        self,
        batch_size,
        vocab_size,
        context_length,
        embedding_len,
        num_heads,
        dropout_prob=0.05,
    ):
        super(DecoderBlockOrig, self).__init__()

        # Attributes
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_len = embedding_len
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.fnn_factor = 4

        # NN Layers
        # Masked MHSA
        self.masked_mhsa_layer = MultiHeadAttention(
            batch_size, context_length, embedding_len, num_heads, 0.2, True
        )
        self.normalisation_mhsa = nn.LayerNorm(embedding_len)

        # Cross attention (Uncomment in encoder-decoder transformer)
        # self.cross_mha_layer = MultiHeadAttention(batch_size, embedding_len, num_heads, 0.2)
        # self.normalisation_cross_mha = nn.LayerNorm(embedding_len)

        # Feed-forward NN
        self.normalisation_fnn = nn.LayerNorm(embedding_len)
        self.fnn = nn.Sequential(
            nn.Linear(embedding_len, embedding_len * self.fnn_factor),
            nn.ReLU(),
            nn.Linear(embedding_len * self.fnn_factor, embedding_len),
            nn.Dropout(self.dropout_prob),
        )

        # Weight initialisation
        self.fnn.apply(self._init_weights)

    def _init_weights(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x):
        # Masked multi-head self attention
        masked_mhsa_pre_norm = self.normalisation_mhsa(x)

        # Peehu debug
        # print("Decoder forward mhsa size : {}, x size : {}".format(masked_mhsa_pre_norm.shape, x.shape))

        masked_mhsa = self.masked_mhsa_layer(
            masked_mhsa_pre_norm, masked_mhsa_pre_norm, masked_mhsa_pre_norm
        )
        masked_mhsa_output = masked_mhsa + x

        # Feedforward NN
        fnn_pre_norm = self.normalisation_fnn(
            masked_mhsa_output
        )  # If cross attention is being used, replace masked_mhsa_output here with cross_mha_output
        fnn = self.fnn(fnn_pre_norm)
        fnn_output = fnn + masked_mhsa_output

        return fnn_output


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
    ):

        # Pass the required arguments to the parent class __init__ method
        super().__init__(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
        )
        
        self.attention = MultiHeadAttention(
            neox_args=neox_args,
            attention_mask_func=gpt2_attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number
        )


# ------------ Test Both Classes ----------- #
if __name__ == "__main__":

    # Define your input dimensions
    batch_size = 16
    vocab_size = 10000
    context_length = 10
    embedding_len = 64
    num_heads = 8
    dropout_prob = 0.05

    neox_args = NeoxArgs(
        attention_config=expand_attention_types([[["global"], 4]], 4),
    )

    initialize_megatron(neox_args=neox_args)
    # Instantiate the DecoderBlockOrig
    decoder_block_orig = DecoderBlockOrig(
        batch_size=batch_size,
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_len=embedding_len,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
    ).to(DEVICE)

    # Instantiate the DecoderBlock
    decoder_block = DecoderBlock(
        neox_args=neox_args,
        attention_mask_func=gpt2_attention_mask_func,
        init_method=get_init_methods(neox_args)[0],
        output_layer_init_method=get_init_methods(neox_args)[1],
        layer_number=1,
    ).to(DEVICE)

    torch.manual_seed(1234)

    # Dummy input tensor
    input_tensor = torch.randn((batch_size, context_length, embedding_len)).to(DEVICE)
    print(f"input tensor shape: {input_tensor.shape}")

    torch.manual_seed(1234)
    # Forward pass through the Orig DecoderBlock
    orig_output_tensor = decoder_block_orig(input_tensor).to(DEVICE)
    # print(f"orignal tensor output: {orig_output_tensor}")
    
    # Dummy Attention Mask
    attention_mask = torch.ones(input_tensor.shape).to(DEVICE)

    # Forward pass through the Orig DecoderBlock
    output_tensor = decoder_block(input_tensor, attention_mask.bool()).to(DEVICE)
    # print(f"tensor output: {output_tensor}")

    # Print the output shape
    print("Output Tensor Shape Orig Decoder Block:", orig_output_tensor.shape)
    print("Output Tensor Shape Decoder Block:", output_tensor.shape)