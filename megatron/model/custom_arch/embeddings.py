
from megatron.model.word_embeddings import Embedding as GPT_Embedding

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
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        use_pos_emb=True,
    ):
        
        # Pass the required arguments to the parent class __init__ method
        super().__init__(
            neox_args,
            hidden_size,
            vocab_size,
            max_sequence_length,
            embedding_dropout_prob,  # Add the appropriate dropout value
            init_method,
            num_tokentypes,
            use_pos_emb,
        )
        
        
class EmbeddingPipe(Embedding):
    """Extends Embedding to forward attention_mask through the pipeline."""

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight

    def forward(self, args):
        assert (
            len(args) == 3
        ), f"Expected 3 arguments (input_ids, position_ids, attention_mask), but got {len(args)}."

        input_ids = args[0]
        position_ids = args[1]
        attention_mask = args[2]
        embeddings = super().forward(input_ids, position_ids)
        return embeddings, attention_mask