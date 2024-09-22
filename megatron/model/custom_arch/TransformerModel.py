"""

from gensim.models import Word2Vec
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import string
import math
from gensim.models import Word2Vec
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import string
import math

from gensim.models import Word2Vec
import torch
import math
import string
word_features = 128
max_input_words = 1024
head_count=8
layer_count=2
epochs = 1
learning_rate = 0.001
word2vec = Word2Vec.load('word2vec.bin')
word_vectors = word2vec.wv
idx_to_key = word_vectors.index_to_key
punctuations = string.punctuation
def clean_text(text):
  cleaned_text = []
  for word in text:
    word = word.lower()
    word = "".join([char for char in word if char not in punctuations])
    cleaned_text.append(word)
  return cleaned_text

def process_input(sentence):
    sentence = sentence.split()
    sentence = clean_text(sentence)
    sen_vec = []
    for word in sentence:
        sen_vec.append(word2vec.wv[word].flatten())
    sen_vec = torch.tensor(sen_vec,dtype=torch.float32)
    sen_vec2 = torch.tensor(sen_vec,dtype=torch.float32)
    sen_vec2 = torch.cat((torch.zeros(1,128), sen_vec[:-1]), dim=0)
    zero_vec = torch.zeros(max_input_words - sen_vec.shape[0],word_features,dtype=torch.float32)
    vec = torch.cat((zero_vec,sen_vec),dim=0)
    vec2 = torch.cat((zero_vec,sen_vec2),dim=0)
    pe = torch.zeros(word_features,max_input_words,dtype=torch.float32)
    for pos in range(word_features):
        for i in range(0,max_input_words,2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/max_input_words)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/max_input_words)))
    pe = pe.transpose(0,1)
    vec = vec + pe 
    vec2 = vec2 + pe
    labels =  [idx_to_key.index(word) for word in sentence]  # Remove the last word from the sentence
    labels = [len(idx_to_key)-1]*(max_input_words-len(labels)) + labels

    return vec, vec2, labels

input = "Before we proceed any further, hear me speak."
input = process_input(input)
def fancy_print_2d_list(matrix, data_width=4):
   data_width = max(data_width + 2, len(str(len(matrix[0]))))
   print(f'{"":>{data_width}} |', end='')
   for col_index in range(len(matrix[0])):
       print(f' {col_index:>{data_width}} |', end='')
   print('\n' + '_' * ((len(matrix[0]) + 1) * (data_width + 3) - 1))
   for row_index, row in enumerate(matrix):
       print(f'{row_index:>{data_width}} |', end='')
       for element in row:
           sign = '+' if element >= 0 else ''
           print(f' {sign}{element:.{data_width - 3}f} |', end='')
       print('\n' + '_' * ((len(matrix[0]) + 1) * (data_width + 3) - 1))

#------------EMBEDDING------------#
class InputEmbedding(nn.Module):
  def __init__(self, context_length, vocab_size, embedding_len, device):
    super(InputEmbedding, self).__init__()
    self.device = device
    self.embedding_layer = nn.Embedding(vocab_size, embedding_len).to(self.device)
    self.pos_embedding_layer = nn.Embedding(context_length, embedding_len).to(self.device)
    self.context_length = context_length

    #Weight initialisation
    torch.nn.init.normal_(self.embedding_layer.weight, mean=0, std=0.02)
    torch.nn.init.normal_(self.pos_embedding_layer.weight, mean=0, std=0.02)

  def forward(self, token_ids, target=False):
    #Token & positional embeddings
    token_embedding = self.embedding_layer(token_ids).to(self.device)
    pos_indices = torch.arange(self.context_length).to(self.device)
    pos_embedding = self.pos_embedding_layer(pos_indices).to(self.device)
    final_embedding = token_embedding + pos_embedding
    return final_embedding

class AttentionHead(nn.Module):
    def __init__(self,word_features,max_input_words,mask = False):
        super(AttentionHead,self).__init__()
        self.word_features = word_features
        self.max_input_words = max_input_words
        self.query = nn.Linear(word_features,word_features,bias=False)
        self.key = nn.Linear(word_features,word_features,bias=False)
        self.value = nn.Linear(word_features,word_features,bias=False)
        self.mask = mask
        self.maskVector = torch.triu(torch.ones(max_input_words,max_input_words,dtype=torch.float32),diagonal=1)

    def scaled_dot_product_attention(self,x):
        self.maskVector = self.maskVector.to(x.device)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        k= torch.transpose(k,0,1)
        matmul_qk = torch.matmul(q,k)
        scaled_attention_logits = matmul_qk / math.sqrt(self.word_features)
        if self.mask:
            scaled_attention_logits = scaled_attention_logits + self.maskVector
        # plt.show()
        output = torch.matmul(scaled_attention_logits,v)
        return output

    def forward(self,x):
        return self.scaled_dot_product_attention(x)

# head = AttentionHead(word_features,max_input_words)
# output = head(pos_encoded)
# output.shape

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward,self).__init__()
        self.Linear1 = nn.Linear(word_features,word_features)
        self.Linear2 = nn.Linear(word_features,word_features)

    def forward(self,x):
        h=self.Linear1(x)
        h=nn.GELU()(h)
        h=self.Linear2(h)
        return h


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward,self).__init__()
        self.Linear1 = nn.Linear(word_features,word_features)
        self.Linear2 = nn.Linear(word_features,word_features)

    def forward(self,x):
        h=self.Linear1(x)
        h=nn.GELU()(h)
        h=self.Linear2(h)
        return h

# ffd = FeedForward()
# output = ffd(output)
# output.shape

class MultiHeadAttention(nn.Module):
    def __init__(self,word_features,max_input_words,head_count,mask = False):
        super(MultiHeadAttention,self).__init__()
        self.word_features = word_features
        self.max_input_words = max_input_words
        self.head_count = head_count
        self.heads = nn.ModuleList()
        self.inweights = nn.ModuleList()
        self.outweights = nn.ModuleList()
        for i in range(head_count):
            self.heads.append(AttentionHead(word_features,max_input_words,mask))
            self.inweights.append(nn.Linear(word_features,word_features,bias=False))
            self.outweights.append(nn.Linear(word_features,word_features,bias=False))

    def forward(self,x,prev_output=None):
        output = torch.zeros(max_input_words,word_features).to(x.device)
        for i in range(self.head_count):
            if prev_output is None:
                output += self.outweights[i](self.heads[i](self.inweights[i](x)))
            else:
                if i<head_count/2:
                    output += self.outweights[i](self.heads[i](self.inweights[i](x)))
                else:
                    output += self.outweights[i](self.heads[i](self.inweights[i](prev_output)))
        return output

# mha = MultiHeadAttention(word_features,max_input_words,head_count)
# output = mha(pos_encoded)
# output.shape

class EncoderLayer(nn.Module):
    def __init__(self,word_features,max_input_words,head_count):
        super(EncoderLayer,self).__init__()
        self.word_features = word_features
        self.head_count = head_count
        self.max_input_words = max_input_words
        self.multi_head_attention = MultiHeadAttention(word_features,max_input_words,head_count)
        self.feed_forward = FeedForward()
        self.normalization = nn.LayerNorm(word_features)

    def forward(self,x):
        multi_head_attention = self.multi_head_attention(x)
        multi_head_attention += x
        multi_head_attention = self.normalization(multi_head_attention)
        feed_forward = self.feed_forward(multi_head_attention)
        feed_forward += multi_head_attention
        feed_forward = self.normalization(feed_forward)
        return feed_forward

# en = EncoderLayer(word_features,max_input_words,head_count)
# output = en(pos_encoded)
# output.shape

class DecoderLayer(nn.Module):
    def __init__(self,word_features,max_input_words,head_count):
        super(DecoderLayer,self).__init__()
        self.word_features = word_features
        self.head_count = head_count
        self.max_input_words = max_input_words
        self.multi_head_attention1 = MultiHeadAttention(word_features,max_input_words,head_count,mask=True)
        self.multi_head_attention2 = MultiHeadAttention(word_features,max_input_words,head_count)
        self.feed_forward = FeedForward()
        self.normalization = nn.LayerNorm(word_features)
    def forward(self,prev,input):
        multi_head_attention1 = self.multi_head_attention1(prev)
        multi_head_attention1 += prev
        multi_head_attention1 = self.normalization(multi_head_attention1)
        multi_head_attention2 = self.multi_head_attention2(input,multi_head_attention1)
        multi_head_attention2 += multi_head_attention1
        multi_head_attention2 = self.normalization(multi_head_attention2)
        feed_forward = self.feed_forward(multi_head_attention2)
        feed_forward += multi_head_attention2
        feed_forward = self.normalization(feed_forward)
        return feed_forward


# en = DecoderLayer(word_features,max_input_words,head_count)
# output = en(torch.zeros(max_input_words,word_features),pos_encoded)
# output.shape

class Encoder(nn.Module):
    def __init__(self,word_features,max_input_words,head_count,layer_count):
        super(Encoder,self).__init__()
        self.word_features = word_features
        self.head_count = head_count
        self.max_input_words = max_input_words
        self.layer_count = layer_count
        self.layers = nn.ModuleList()
        for i in range(layer_count):
            self.layers.append(EncoderLayer(word_features,max_input_words,head_count))
    def forward(self,x):
        for i in range(self.layer_count):
            x = self.layers[i](x)
        return x

# en = Encoder(word_features,max_input_words,head_count,layer_count)
# output = en(pos_encoded)
# output.shape

class Decoder(nn.Module):
    def __init__(self,word_features,max_input_words,head_count,layer_count):
        super(Decoder,self).__init__()
        self.word_features = word_features
        self.head_count = head_count
        self.max_input_words = max_input_words
        self.layer_count = layer_count
        self.layers = nn.ModuleList()
        for i in range(layer_count):
            self.layers.append(DecoderLayer(word_features,max_input_words,head_count))
    def forward(self,prev,x):
        for i in range(self.layer_count):
            x = self.layers[i](prev,x)
        return x

# dec = Decoder(word_features,max_input_words,head_count,layer_count)
# output = dec(torch.zeros(max_input_words,word_features),pos_encoded)
# output.shape

class Transformer(nn.Module):
    def __init__(self,word_features,max_input_words,head_count,layer_count, vocab_size, device):
        super(Transformer,self).__init__()
        self.word_features = word_features
        self.head_count = head_count
        self.max_input_words = max_input_words
        self.layer_count = layer_count
        self.vocab_size = vocab_size
        self.device = device
        # self.encoder = Encoder(word_features,max_input_words,head_count,layer_count)
        self.decoder = Decoder(word_features,max_input_words,head_count,layer_count)
        self.final = nn.Linear(word_features,len(idx_to_key)+1)
        self.softmax = nn.Softmax(dim=1)
        self.inputEmbedder = InputEmbedding(max_input_words, vocab_size, word_features, device)
    def forward(self,x, targets=None):
        # encoder_output = self.encoder(x)
        x_embed = self.inputEmbedder(x)
        decoder_output = self.decoder(prev,encoder_output)
        output = self.final(decoder_output)
        output = self.softmax(output)
        return output




"""

# Import important libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import PretrainedConfig, PreTrainedModel
from typing import List

# Define hyperparameters - These are the hyperparameters that have given me the best result yet

# batch_size = 8  #Num of sentences being processed in parallel
# context_length = 1024  #Num of tokens processed at a time (how much context is there behind understanding each token)
# embedding_len = 128 #Each token is converted into an embedding_len dimensional tensor once it undergoes embedding
# num_heads = 8 #Num of heads that the embedding matrices will be split in while computing attention
# num_encoder_blocks = 1
# num_decoder_blocks = 2
learning_rate = 5e-5
max_iterations = 150000  # Num of iterations for which model is trained
eval_interval = 500  # Num of iterations after which validation loss is computed (during model training)
val_iterations = 200
checkpoint_interval = 10000  # Num of iterations after which a checkpoint is created
num_generated_tokens = 10000  # Num of tokens generated from a trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


"""
#Download dataset (I have another public repo just for datasets I used for this project)
!wget 'https://raw.githubusercontent.com/bl0nder/makespeare_datasets/main/shakespeare_input.txt'

#Read the dataset
with open('shakespeare_input.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

#------------TOKENISATION------------#
#Character-Level Tokenization
char_list = sorted(list(set(input_text)))
vocab_size = len(char_list)

char_to_token = {}
token_to_char = {}

for i,c in enumerate(char_list):
  char_to_token[c] = i
  token_to_char[i] = c

#Function to encode string into tokens
def encode(string):
  tokens = []
  for c in string:
    tokens.append(char_to_token[c])
  return tokens

#Function to decode tokens into corresponding characters
def decode(tokens):
  chars = []
  for i in tokens:
    chars.append(token_to_char[i])
  return ''.join(chars)

#Convert token array to tensor for further processing
token_ids = torch.tensor(encode(input_text))

#Train/val split
train_idx = int(len(token_ids)*0.9)
train_data = token_ids[0:train_idx]
val_data = token_ids[train_idx:]

"""


# ------------MINI-BATCH SELECTION------------#
def minibatch(train_data, val_data, context_length, batch_size, train=True):

    # Selecting whether to sample from training or validation data
    if train:
        data = train_data
    else:
        data = val_data

    # Random index to pick minibatch from
    ind = torch.randint(0, len(data) - context_length, size=(batch_size,))

    # Create minibatch
    x_batch = torch.stack([data[i : i + context_length] for i in ind])  # Tokens
    y_batch = torch.stack(
        [data[i + 1 : i + context_length + 1] for i in ind]
    )  # Next tokens in sentence

    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    return x_batch, y_batch


# ------------EMBEDDING------------ #
class InputEmbedding(nn.Module):
    def __init__(self, context_length, vocab_size, embedding_len):
        super(InputEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_len).to(DEVICE)
        self.pos_embedding_layer = nn.Embedding(context_length, embedding_len).to(
            DEVICE
        )
        self.context_length = context_length

        # Weight initialisation
        torch.nn.init.normal_(self.embedding_layer.weight, mean=0, std=0.02)
        torch.nn.init.normal_(self.pos_embedding_layer.weight, mean=0, std=0.02)


def forward(self, token_ids, target=False):
    # Token & positional embeddings
    token_embedding = self.embedding_layer(token_ids).to(DEVICE)
    pos_indices = torch.arange(self.context_length).to(DEVICE)
    pos_embedding = self.pos_embedding_layer(pos_indices).to(DEVICE)
    final_embedding = token_embedding + pos_embedding
    return final_embedding


# ------------ATTENTION!------------ #
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        context_length,
        embedding_len,
        num_heads,
        dropout_prob=0.2,
        attention_mask=False,
    ):

        super(MultiHeadAttention, self).__init__()

        self.batch_size = batch_size
        self.embedding_len = embedding_len
        self.num_heads = num_heads
        self.head_dim = embedding_len // num_heads
        self.attention_mask = attention_mask
        self.context_length = context_length

        # Embedding length needs to be divisible by # of heads
        assert self.head_dim == float(
            embedding_len / num_heads
        ), "embedding_len must be divisible by num_heads"

        # Linear layers to compute Wq, Wk, Wv
        self.W_q = nn.Linear(embedding_len, embedding_len, bias=False).to(DEVICE)
        self.W_k = nn.Linear(embedding_len, embedding_len, bias=False).to(DEVICE)
        self.W_v = nn.Linear(embedding_len, embedding_len, bias=False).to(DEVICE)

        # Linear layer + Dropout for output
        self.output = nn.Linear(embedding_len, embedding_len).to(DEVICE)
        self.output_dropout = nn.Dropout(dropout_prob)

        # Weight initialisation for nn layers
        torch.nn.init.normal_(self.W_q.weight, mean=0, std=0.02)
        torch.nn.init.normal_(self.W_k.weight, mean=0, std=0.02)
        torch.nn.init.normal_(self.W_v.weight, mean=0, std=0.02)

    def forward(self, v, k, q):

        # Compute Values, Keys and Queries
        V = self.W_v(v).to(DEVICE)
        K = self.W_k(k).to(DEVICE)
        Q = self.W_q(q).to(DEVICE)

        # Peehu debug
        # print("Forward size of Q : {}, of K : {}, of V : {}".format(Q.shape, K.shape, V.shape))

        # Split into num_heads heads for multi-head processing
        V_split, Q_split, K_split = self.split(V, Q, K)

        # Compute scaled dot-product attention
        attention, attention_weights = self.scaled_dot_product_attention(
            V_split, K_split, Q_split
        )

        # Concatenate heads
        attention_concat = self.concat_heads(attention)

        # Pass attention through linear layer
        mha_output = self.output_dropout(self.output(attention_concat))
        return mha_output

    def split(self, V, Q, K):

        # Splitting values, keys and queries into num_head heads
        V_split = torch.stack(torch.split(V, self.head_dim, dim=2), dim=1)
        Q_split = torch.stack(torch.split(Q, self.head_dim, dim=2), dim=1)
        K_split = torch.stack(torch.split(K, self.head_dim, dim=2), dim=1)

        return V_split, Q_split, K_split

    def concat_heads(self, attention):

        # This is better understood with a diagram so here it is:
        # [[1 2 3
        #  1 2 3    <- Head #1
        #  1 2 3]
        # [4 5 6
        #  4 5 6    <- Head #2
        #  4 5 6]
        # [7 8 9
        #  7 8 9    <- Head #3
        #  7 8 9]]
        # We wanna transpose the matrix such that we get:
        # [[1 2 3
        #  4 5 6  <- First row of each head
        #  7 8 9]
        # [1 2 3
        #  4 5 6  <- Second row of each head
        #  7 8 9]
        # [1 2 3
        #  4 5 6  <- Third row of each head
        #  7 8 9]]

        attention_concat = attention.transpose(1, 2)

        # Now we just wanna 'stretch out' the heads to get the concatenated attention matrix:
        # [[1 2 3 4 5 6 7 8 9] <- First row matrix stretched out
        # [1 2 3 4 5 6 7 8 9] <- Second row matrix stretched out
        # [1 2 3 4 5 6 7 8 9]] <- Third row matrix stretched out

        attention_concat = attention_concat.reshape(
            self.batch_size, self.context_length, -1
        )
        return attention_concat

    def scaled_dot_product_attention(self, V, K, Q):

        # Attention = Softmax(QK.T/sqrt(d_k))*V

        K_T = torch.transpose(K, -2, -1)
        QK = torch.einsum("abij, abjk -> abik", [Q, K_T])

        # Look-ahead mask
        if self.attention_mask == True:
            # bat_size = QK.shape[0]
            mask = (
                torch.tril(torch.ones((self.context_length, self.context_length)))
                .expand(self.num_heads, self.context_length, self.context_length)
                .to(DEVICE)
            )
        mask = mask.expand(
            self.batch_size, self.num_heads, self.context_length, self.context_length
        )

        # Peehu debug
        # print("Size of QK : {}, of mask : {}, of Q : {}, of K : {}, of V : {}".format(QK.shape, mask.shape, Q.shape, K.shape, V.shape))
        # print("Size of mask : {}".format(mask.shape))

        QK = QK.masked_fill(mask == 0, float("-inf"))

        d_k = K.shape[-1]
        product = QK / np.sqrt(d_k)

        temp = nn.Softmax(dim=-1)
        attention_weights = temp(product)
        attention = torch.einsum("abij, abjk -> abik", [attention_weights, V])

        return attention, attention_weights


# ------------ENCODER------------#
class EncoderBlock(nn.Module):
    def __init__(
        self,
        batch_size,
        num_heads,
        vocab_size,
        context_length,
        embedding_len,
        dropout_prob=0.05,
    ):

        super(EncoderBlock, self).__init__()

        # Attributes
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_len = embedding_len
        self.dropout_prob = dropout_prob
        self.fnn_factor = 4

        # Required NN layers
        self.multi_head_self_attention_layer = MultiHeadAttention(
            batch_size, self.embedding_len, num_heads
        ).to(DEVICE)
        self.normalisation_mhsa = nn.LayerNorm(embedding_len).to(DEVICE)
        self.normalisation_fnn = nn.LayerNorm(
            (batch_size, self.context_length, self.embedding_len)
        ).to(DEVICE)
        self.fnn = nn.Sequential(
            nn.Linear(embedding_len, embedding_len * self.fnn_factor),
            nn.ReLU(),
            nn.Linear(embedding_len * self.fnn_factor, embedding_len),
            nn.Dropout(self.dropout_prob),
        ).to(DEVICE)

        # Weight initialisation
        self.fnn.apply(self._init_weights)


def _init_weights(self, module):
    if type(module) == nn.Linear:
        torch.nn.init.normal_(module.weight, mean=0, std=0.02)


def forward(self, x, batch_size, num_heads, verbose=False):

    # Add & Pre-Norm
    mhsa_pre_norm = self.normalisation_mhsa(
        x
    )  # Even though the original paper uses normalisation after computing self-attention, pre-normalisation may produce better results (and it did in this case)
    mhsa = self.multi_head_self_attention_layer(
        mhsa_pre_norm, mhsa_pre_norm, mhsa_pre_norm
    )
    mhsa_output = mhsa + x

    # Feed-forward NN
    fnn_pre_norm = self.normalisation_fnn(mhsa_output)
    fnn = self.fnn(fnn_pre_norm)
    fnn_output = fnn + mhsa_output

    return fnn_output



# #harshits's rearrangement of normalization
# def forward(self, x, batch_size, num_heads, verbose=False):
#     print("encoder mod")
#     y = self.multi_head_self_attention_layer(x,x,x)
#     y = self.normalisation_mhsa(y)
#     x = x + y
#     y = self.fnn(x)
#     y = self.normalisation_fnn(y)
#     return x + y

# Encoder class
class Encoder(nn.Module):
    def __init__(
        self,
        num_encoder_blocks,
        batch_size,
        vocab_size,
        context_length,
        embedding_len,
        num_heads,
        dropout_prob=0.05,
    ):

        super(Encoder, self).__init__()

        # Attributes
        self.num_encoder_blocks = num_encoder_blocks
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.encoder_blocks = []
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_len = embedding_len
        self.dropout_prob = dropout_prob

        # List of encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    batch_size,
                    num_heads,
                    vocab_size,
                    context_length,
                    embedding_len,
                    self.dropout_prob,
                )
                for i in range(self.num_encoder_blocks)
            ]
        )


def forward(self, x):
    encoder_input = x
    for i, block in enumerate(self.encoder_blocks):
        encoder_input = block(encoder_input, self.batch_size, self.num_heads).to(DEVICE)

    encoder_output = encoder_input
    return encoder_output


# ------------DECODER------------#
class DecoderBlock(nn.Module):
    def __init__(
        self,
        batch_size,
        vocab_size,
        context_length,
        embedding_len,
        num_heads,
        dropout_prob=0.05,
    ):
        super(DecoderBlock, self).__init__()

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


def forward(self, x, q_cross, k_cross):
    # Masked multi-head self attention
    masked_mhsa_pre_norm = self.normalisation_mhsa(x)

    # Peehu debug
    # print("Decoder forward mhsa size : {}, x size : {}".format(masked_mhsa_pre_norm.shape, x.shape))

    masked_mhsa = self.masked_mhsa_layer(
        masked_mhsa_pre_norm, masked_mhsa_pre_norm, masked_mhsa_pre_norm
    )
    masked_mhsa_output = masked_mhsa + x

    # Multi-head cross attention - Uncomment the following 3 lines if using encoder-decoder transformer. Redundant in decoder-only model (such as this one) since there is no encoder output to calculate cross attention with
    # cross_mha_pre_norm = self.normalisation_cross_mha(masked_mhsa_output)
    # cross_mha = self.cross_mha_layer(cross_mha_pre_norm, k_cross, q_cross)
    # cross_mha_output = cross_mha + masked_mhsa_output

    # Feedforward NN
    fnn_pre_norm = self.normalisation_fnn(
        masked_mhsa_output
    )  # If cross attention is being used, replace masked_mhsa_output here with cross_mha_output
    fnn = self.fnn(fnn_pre_norm)
    fnn_output = fnn + masked_mhsa_output

    return fnn_output


# #harshits's rearrangement of normalization
# def forward(self, x, x, q_cross, k_cross):
#     print("decoder mod")
#     y = self.multi_head_self_attention_layer(x,x,x)
#     y = self.normalisation_mhsa(y)
#     x = x + y
#     y = self.fnn(x)
#     y = self.normalisation_fnn(y)
#     return x + y


class Decoder(nn.Module):
    def __init__(
        self,
        num_decoder_blocks,
        batch_size,
        vocab_size,
        context_length,
        embedding_len,
        num_heads,
        dropout_prob=0.05,
    ):
        super(Decoder, self).__init__()

        # Attributes
        self.num_decoder_blocks = num_decoder_blocks
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_len = embedding_len
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # List of decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    batch_size,
                    vocab_size,
                    context_length,
                    embedding_len,
                    num_heads,
                    dropout_prob,
                )
                for i in range(num_decoder_blocks)
            ]
        )


def forward(self, x):
    # Loop through all decoder blocks and process inputs sequentially (output of a block is input to the next)
    decoder_input = x

    # Peehu debug
    # print("Full Decoder input x size : {}".format(decoder_input.shape))

    for i, block in enumerate(self.decoder_blocks):
        decoder_input = block(decoder_input, x, x)

    decoder_output = decoder_input
    return decoder_output


class TransformerConfig(PretrainedConfig):
    model_type = "custom_arch"


def __init__(
    self,
    contextLen=1024,
    embeddingLen=128,
    numHeads=8,
    numEncoderBlocks=1,
    numDecoderBlocks=2,
    vocabSize=19212,
    batchSize=8,
    bos_token_id=1,
    eos_token_id=2,
    **kwargs,
):

    self.contextLen = contextLen
    self.embeddingLen = embeddingLen
    self.numHeads = numHeads
    self.numEncoderBlocks = numEncoderBlocks
    self.numDecoderBlocks = numDecoderBlocks
    self.vocabSize = vocabSize
    self.batchSize = batchSize
    super().__init__(**kwargs, bos_token_id=bos_token_id, eos_token_id=eos_token_id)


# ------------TRANSFORMER------------#
class Transformer(PreTrainedModel):

    config_class = TransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocabSize = config.vocabSize
        self.batchSize = config.batchSize
        self.contextLength = config.contextLen
        self.embeddingLen = config.embeddingLen
        self.numHeads = config.numHeads
        self.numDecoderBlocks = config.numDecoderBlocks
        self.numEncoderBlocks = config.numEncoderBlocks

        self.decoder = Decoder(
            self.numDecoderBlocks,
            self.batchSize,
            self.vocabSize,
            self.contextLength,
            self.embeddingLen,
            self.numHeads,
            0.2,
        )

        # NN Layers
        self.normalisation = nn.LayerNorm(self.embeddingLen)  # final layer norm
        self.linear = nn.Linear(self.embeddingLen, self.vocabSize)

        # Token embedding
        self.input_embedding = InputEmbedding(
            self.contextLength, self.vocabSize, self.embeddingLen
        )

        # Weight Initialisation
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif type(module) == nn.Embedding:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def generate(self, input, max_length=None):
    out = super().generate()
    return out


def forward(self, input_ids=None, attention_mask=None, labels=None):
    # Input embeddings

    x_embeddings = self.input_embedding(input_ids)

    # Peehu debug
    # print("Input ids size : {}, x_embedd size : {}".format(input_ids.shape, x_embeddings.shape))

    # Uncomment the following line if using an encoder-decoder model
    # encoder_output = self.encoder(x_embeddings)
    decoder_output = self.decoder(
        x_embeddings
    )  # Replace x_embeddings with encoder_output if using an encoder-decoder model
    normalised_decoder_output = self.normalisation(decoder_output)
    logits = self.linear(normalised_decoder_output)

    # If targets are given, compute loss
    if labels is None:
        return {"logits": logits}
        # loss = None
    else:

        # labels = torch.nn.functional.one_hot(labels, num_classes=self.vocabSize).type(torch.FloatTensor).to(DEVICE)

        # logits = logits.reshape(self.batchSize, self.contextLength, -1)
        # labels = labels.reshape(self.batchSize, self.contextLength)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = torch.swapaxes(shift_logits, 1, 2)

        # Peehu debug
        # print("Logits shape : ", shift_logits.shape)
        # print("Labels shape : ", shift_labels.shape)

        loss = F.cross_entropy(shift_logits, shift_labels)
        return {"loss": loss, "logits": logits}

        # return logits, loss

    # def generate(self, x, max_new_tokens):
    #   for i in range(max_new_tokens):
    #     x_latest = x[:, -context_length:]
    #     logits, loss = self(x_latest)
    #     logits = logits[:, -1, :]
    #     probs = F.softmax(logits, dim=-1)
    #     x_next = torch.multinomial(probs, num_samples=1).reshape(batch_size, 1)
    #     print(token_to_char[x_next[-1].item()], end='')
    #     x = torch.cat((x, x_next), dim=1)
    #   return x


# ------------TRAINING------------#
# Function to compute validation loss
@torch.no_grad()
def val_loss(model, val_iterations):
    with torch.no_grad():
        out = {"train": 0, "val": 0}
    model.eval()

    for i in range(2):
        for j in range(val_iterations):
            if i == 0:
                x, y = minibatch(train_data, val_data, context_length, batch_size)
            else:
                x, y = minibatch(
                    train_data, val_data, context_length, batch_size, train=False
                )

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits, cross_entropy_loss = model(x, y)

            if i == 0:
                out["train"] += cross_entropy_loss
            else:
                out["val"] += cross_entropy_loss

    out["train"] /= val_iterations
    out["val"] /= val_iterations

    model.train()
    return out


if __name__ == "__main__":
    transformer = Transformer().to(DEVICE)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
    print(
        sum(p.numel() for p in transformer.parameters()) / 1e6, "M parameters"
    )  # Number of params in model

    # Training loop
    for i in range(max_iterations):

        # After every eval_interval iterations, compute validation loss
        if (i + 1) % eval_interval == 0:
            losses = val_loss(transformer, val_iterations)
            print(f"step {i+1}: train loss {losses['train']}, val loss {losses['val']}")

        # Every checkpoint_interval iterations, create a checkpoint for the model, i.e, save the model state dictionary (along with other info if you want) somewhere
        if (i + 1) % checkpoint_interval == 0:
            checkpoint = {
                "iterations": i + 1,
                "num_encoder_blocks": num_encoder_blocks,
                "num_decoder_blocks": num_decoder_blocks,
                "state_dict": transformer.state_dict(),  # Most important thing to save
            }
            torch.save(
                checkpoint,
                f"models/checkpoint_ctx{context_length}_iter{i+1}_character_encoding.pth",
            )

        # Get minibatch of training data and compute loss
        x, y = minibatch(train_data, val_data, context_length, batch_size, True)
        logits, loss = transformer(x, y)

        # Learn
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ------------TEXT GENERATION------------#
    # Using a pre-trained model by loading a checkpoint
    model = Transformer().to(DEVICE)
    state_dict = torch.load(
        "models/checkpoint_ctx256_iter150000_character_encoding.pth"
    )  # Load saved model

    # When I trained the model, I had an embedding layer in the Decoder class instead of the Transformer class, which I have changed since then. In order for the model to work, 2 of the keys need to be renamed.
    # Comment the following 4 lines if another model is trained.
    state_dict["state_dict"]["input_embedding.embedding_layer.weight"] = state_dict[
        "state_dict"
    ]["decoder.input_embedding.embedding_layer.weight"]
    state_dict["state_dict"]["input_embedding.pos_embedding_layer.weight"] = state_dict[
        "state_dict"
    ]["decoder.input_embedding.pos_embedding_layer.weight"]
    del state_dict["state_dict"]["decoder.input_embedding.embedding_layer.weight"]
    del state_dict["state_dict"]["decoder.input_embedding.pos_embedding_layer.weight"]

    model.load_state_dict(state_dict["state_dict"])  # Load state dictionary into model

    # Generating Shakespearean text
    context = torch.ones((batch_size, context_length), dtype=torch.long, device=DEVICE)
    context *= 8  # Token for full-stop
    gen_output = decode(
        model.generate(context, max_new_tokens=num_generated_tokens)[0].tolist()
    )