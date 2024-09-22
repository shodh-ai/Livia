import sys
import torch
import torch.nn as nn

sys.path.append("/nlsasfs/home/shodhlab/ptarun/InferQ/JayDeep/gpt-neox-custom")

from megatron.model.init_functions import get_init_methods
from megatron.model.word_embeddings import Embedding as GPT_Embedding

from megatron.initialize import initialize_megatron
from neox_args import NeoxArgs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------- EMBEDDING CUSTOM ----------- #
class InputEmbedding(nn.Module):
    def __init__(self, context_length, vocab_size, embedding_len):
        super(InputEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_len).to(DEVICE)
        self.pos_embedding_layer = nn.Embedding(context_length, embedding_len).to(
            DEVICE
        )
        self.context_length = context_length

        # Weight initialisation
        torch.manual_seed(1234)
        torch.nn.init.normal_(self.word_embeddings.weight, mean=0, std=0.02)
        print(f"Embedding word weights: {self.word_embeddings.weight}")
        torch.manual_seed(1234)
        torch.nn.init.normal_(self.pos_embedding_layer.weight, mean=0, std=0.02)
        print(f"Input Embedding pos weights: {self.pos_embedding_layer.weight}")

    def forward(self, token_ids, target=False):
        # Token & positional embeddings
        token_embedding = self.word_embeddings(token_ids).to(DEVICE)
        torch.manual_seed(1234)
        pos_indices = torch.arange(self.context_length).to(DEVICE)
        torch.manual_seed(1234)
        pos_embedding = self.pos_embedding_layer(pos_indices).to(DEVICE)
        # print(f'Input Embedding Word Embeddings init: {token_embedding}')
        # print(f'Input Embedding pos  Embeddings init: {pos_embedding}')
        final_embedding = token_embedding + pos_embedding
        return final_embedding


# ----------- PARALLEL EMBEDDING CUSTOM ----------- #


class Embedding(GPT_Embedding):
    """
    Language Model Embeddings
    Arguments:
        context_length:
        vocab_size (int): size of the dictionary of embeddings
        embedding_len: (int): the size of each embedding vector
    """

    def __init__(
        self,
        neox_args,
        context_length,
        vocab_size,
        embedding_len,
        init_method,
    ):
        # Pass the required arguments to the parent class __init__ method
        super().__init__(
            neox_args=neox_args,
            hidden_size=embedding_len,
            vocab_size=vocab_size,
            max_sequence_length=context_length,
            embedding_dropout_prob=0,  # Add the appropriate dropout value
            init_method=init_method,
        )


# ------------ Test Both Classes ----------- #

if __name__ == "__main__":

    # torch.cuda.manual_seed(1234)
    torch.manual_seed(1234)

    # Dummy input parameters
    context_length = 10
    vocab_size = 1000
    embedding_len = 128

    neox_args = NeoxArgs(
        mup_embedding_mult=None,
        mup_rp_embedding_mult=1,
        init_method="normal",
        init_method_std=0.02,
        use_mup=False,
        mup_init_scale=0.5,
        pos_emb="learned",
        opt_pos_emb_offset=None,
        use_bnb_optimizer=False,
        output_layer_init_method="normal",
    )

    initialize_megatron(neox_args=neox_args)

    # Dummy input token_ids
    token_ids = torch.randint(0, vocab_size, (context_length,)).to(DEVICE)
    print(f"token_ids shape: {token_ids.shape}")
    torch.manual_seed(1234)

    # Instantiate InputEmbedding
    input_embedding = InputEmbedding(context_length, vocab_size, embedding_len).to(
        DEVICE
    )
    print(f"input_embedding: {input_embedding}")
    torch.manual_seed(1234)

    # Forward pass through InputEmbedding
    input_embedding_output = input_embedding(token_ids)
    print(f"input_embedding_output shape: {input_embedding_output.shape}")

    torch.manual_seed(1234)

    # Instantiate Embedding
    embedding = Embedding(
        neox_args,
        context_length,
        vocab_size,
        embedding_len,
        init_method=get_init_methods(neox_args)[0],
    ).to(DEVICE)
    
    print(f"embedding: {embedding}")
    torch.manual_seed(1234)
    
    # pos ids
    position_ids = torch.arange(context_length).to(DEVICE)

    # Forward pass through Embedding
    torch.manual_seed(1234)
    embedding_output = embedding(token_ids, position_ids)
    
    print(f"embedding output shape: {embedding_output.shape}")

    # Compare outputs
    print("InputEmbedding output shape:", input_embedding_output.shape)
    print("Embedding output shape:", embedding_output.shape)

    # You can also compare the actual values if needed
    print("InputEmbedding output:", input_embedding_output)
    print("Embedding output:", embedding_output)