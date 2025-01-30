import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalSelfAttention(nn.Module):

    def __init__(self, n_sequence, n_dim, n_head):
        super().__init__()
        assert n_dim % n_head == 0
        self.Wqkv = nn.Linear(n_dim, 3 * n_dim, bias=False)
        self.Wo = nn.Linear(n_dim, n_dim, bias=True)
        self.register_buffer("bias", torch.tril(torch.ones(n_sequence, n_sequence)).view(1, 1, n_sequence, n_sequence))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v  = self.Wqkv(x).split(C, dim=2) # in=[B, T, n_dim], out=3 * [B, T, C]
        
        k = k.view(-1, T, self.n_head, C // self.n_head).transpose(1, 2) # out=(B, n_head, T, C // self.n_head)
        q = q.view(-1, T, self.n_head, C // self.n_head).transpose(1, 2) # out=(B, n_head, T, C // self.n_head)
        v = v.view(-1, T, self.n_head, C // self.n_head).transpose(1, 2) # out=(B, n_head, T, C // self.n_head)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(C // self.n_head)) # out=(B, n_head, T, T)
        att = att.masked_fill(self.bias == 0, float('-inf')) # out=(B, n_head, T, T)
        att = F.softmax(att, dim=-1) # out=(B, n_head, T, T)
        att = F.dropout(att) # out=(B, n_head, T, T)
        
        y = att @ v # out=(B, n_head, T, C // self.n_head)
        y = y.transpose(1, 2) # out=(B, T, n_head, C // self.n_head)
        y = y.contiguous().view(B, T, C) # out=(B, T, C)
        y = self.Wo(y) # out=(B, T, n_dim)
        y = F.dropout(y, inplace=True) # out=(B, T, C)
        return y 

class FakeDecoder(nn.Module):
    def __init__(self, n_sequence, n_dim, n_head, vocab_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, n_dim)
        self.attention_layer = CausalSelfAttention(n_sequence, n_dim, n_head)
        self.last_layer = nn.Linear(n_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.attention_layer(x)
        return self.last_layer(x)

if __name__ == "__main__":
    n_sequence = 10  # A sequence of 10 tokens/letters.
    n_dim = 8  # Each token is 8-d vector.
    n_head = 2  # Multi-heads attention.
    
    # Next letter prediction.
    sentence = 'HELLO WOLRD! HOW ARE YOU DOING TODAY?'
    vocab = set(sentence[::])
    letter_to_index = {l:i for i, l in enumerate(vocab)}

    x_ids_list = []
    y_ids_list = []
    for window_i in range(0, len(sentence)-10):
        every_x_letters = sentence[window_i : window_i + n_sequence + 1]
        x_ids_list.append([letter_to_index[l] for l in every_x_letters[:-1]])
        # Target sequence shifted right as the label. Work together with Mask.
        y_ids_list.append([letter_to_index[l] for l in every_x_letters[1:]])
    x_ids = torch.tensor(x_ids_list)
    y_ids = torch.tensor(y_ids_list)
    y_ids = y_ids.view(-1)
    
    model = FakeDecoder(n_sequence, n_dim, n_head, len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    
    n_epoch = 10
    for _ in range(n_epoch):
        optimizer.zero_grad()
        y_logits = model(x_ids)
        loss = F.cross_entropy(y_logits.view(-1, len(vocab)), y_ids)
        optimizer.step()
        print(f"epoch {_}: loss={loss:.3f}")
        
    # Teacher Forcing