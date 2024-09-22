import torch
import torch.nn as nn
from torchinfo import summary

class GPTNeoX(nn.Module):
    def __init__(self, config):
        super(GPTNeoX, self).__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.position_embedding = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])

        self.layers = nn.ModuleList([
            TransformerLayer(config['hidden_size'], config['num_attention_heads'], config['hidden_size'] * 4, config['num_attention_groups'])
            for _ in range(config['num_layers'])
        ])
        self.ln_f = nn.LayerNorm(config['hidden_size'])
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def forward(self, input_ids):
        positions = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, ffn_hidden_size, num_attention_groups):
        super(TransformerLayer, self).__init__()
        self.self_attention = GroupedQueryAttention(hidden_size, num_attention_heads, num_attention_groups)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size, hidden_size)
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        attn_output, _ = self.self_attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.ln1(hidden_states + attn_output)
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ln2(hidden_states + ffn_output)
        return hidden_states

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_attention_groups):
        super(GroupedQueryAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.group_size = num_attention_heads // num_attention_groups
        head_dim = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, head_dim * num_attention_groups)
        self.value = nn.Linear(hidden_size, head_dim * num_attention_groups)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value):
        batch_size, seq_length, hidden_size = query.size()
        head_dim = hidden_size // self.num_attention_heads

        query = self.query(query).view(batch_size, seq_length, self.num_attention_heads, head_dim)
        key = self.key(key).view(batch_size, seq_length, self.num_attention_groups, head_dim)
        value = self.value(value).view(batch_size, seq_length, self.num_attention_groups, head_dim)

        query = query.view(batch_size, seq_length, self.num_attention_groups, self.group_size, head_dim)
        key = key.unsqueeze(3).expand(-1, -1, -1, self.group_size, -1)
        value = value.unsqueeze(3).expand(-1, -1, -1, self.group_size, -1)

        attn_output = []
        for group in range(self.num_attention_groups):
            q = query[:, :, group].reshape(batch_size, seq_length, -1)
            k = key[:, :, group].reshape(batch_size, seq_length, -1)
            v = value[:, :, group].reshape(batch_size, seq_length, -1)

            scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
            attn_weights = nn.functional.softmax(scores, dim=-1)
            attn_output_group = torch.matmul(attn_weights, v)
            attn_output.append(attn_output_group)

        attn_output = torch.cat(attn_output, dim=-1)
        attn_output = self.out(attn_output.view(batch_size, seq_length, hidden_size))

        return attn_output, attn_weights

# Configuration dictionary
config = {
    'vocab_size': 38400,
    'hidden_size': 2048,
    'num_layers': 24,
    'num_attention_heads': 32,
    'max_position_embeddings': 2048,
    'num_attention_groups': 4  # Number of attention groups for GQA
}

# Instantiate the model
model = GPTNeoX(config)

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Use a dummy input to get the summary
input_ids = torch.randint(0, config['vocab_size'], (1, config['max_position_embeddings'])).to(device)
summary(model, input_data=(input_ids,))
