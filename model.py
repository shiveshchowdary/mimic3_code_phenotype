import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F




class TimeSinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeSinusoidalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, input_hours):
        position = torch.arange(0, self.embedding_dim, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, dtype=torch.float32) * -(np.log(10000.0) / self.embedding_dim))
        div_term = div_term.to(DEVICE)
        
        sin_terms = torch.sin(input_hours.unsqueeze(-1) * div_term).to(DEVICE)
        cos_terms = torch.cos(input_hours.unsqueeze(-1) * div_term).to(DEVICE)


        sinusoidal_embedding = torch.empty(input_hours.size(0),input_hours.size(1),self.embedding_dim, dtype=torch.float32)
        sinusoidal_embedding[:, :, 0::2] = sin_terms
        sinusoidal_embedding[:, :, 1::2] = cos_terms
        return sinusoidal_embedding.to(DEVICE)

class ContinuousValueEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = self.embedding(x.unsqueeze(2))
        out = self.tanh(out)
        return out


class VariableEmbedding(nn.Module):
    def __init__(self, d_model, num_variables):
        super().__init__()
        self.embedding = nn.Embedding(num_variables+1, d_model)
        
    def forward(self, x):
        return self.embedding(x)
    
    

class Embedding(nn.Module):
    def __init__(self, d_model, num_variables, sinusoidal):
        super().__init__()
        self.cvs_value = ContinuousValueEmbedding(d_model)
        if sinusoidal:
            self.cvs_time = TimeSinusoidalEmbedding(d_model)
        else:
            self.cvs_time = ContinuousValueEmbedding(d_model)
        self.var_embed = VariableEmbedding(d_model, num_variables)
    def forward(self, encoder_input):
        time = encoder_input[0]
        variable = encoder_input[1]
        value = encoder_input[2]
        embed = self.cvs_time(time) + self.cvs_value(value) + self.var_embed(variable)
        return embed

class Attention(nn.Module):
    def __init__(self, d_model, d, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.d = d
        self.Q = nn.Linear(d_model, d)
        self.K = nn.Linear(d_model, d)
        self.V = nn.Linear(d_model, d)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x, mask): 
        q = self.Q(x) 
        k = self.K(x)
        v = self.V(x) 
        weights = q@k.transpose(-2,-1)*k.shape[-1]**(-0.5) 
        weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1) 
        self.dropout(weights)
        out = weights @ v
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_model, d_model//n_heads) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads*(d_model//n_heads), d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        out = self.W1(x)
        out = F.relu(out)
        out = self.dropout(self.W2(out))
        return out

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.multi_attention = MultiHeadAttention(d_model, n_heads)
        self.ffb = FeedForwardBlock(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        out = self.multi_attention(x, mask)
        out1 = x + self.ln2(out)
        out2 = self.ffb(out1)
        out = out1 + self.ln2(out2)
        return out

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_variables , N, sinusoidal):
        super().__init__()
        self.embedding = Embedding(d_model, num_variables, sinusoidal)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(N)])
        self.N = N
    
    def forward(self, encoder_input, mask):
        time = encoder_input[0]
        variable = encoder_input[1]
        value = encoder_input[2]
        x = self.embedding((time, variable, value))
        for block in self.encoder_blocks:
            x = block(x, mask)
        return x

class FusionSelfAttention(nn.Module):
    def __init__(self, d_model, dropout = 0.2):
        super().__init__()
        self.Wa = nn.Linear(d_model, d_model)
        self.Ua = nn.Linear(d_model, d_model)
        self.Va = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, out, mask):
        q = out.unsqueeze(2) 
        k = out.unsqueeze(1) 
        v = out 
        a = F.tanh(self.Wa(q) + self.Ua(k)) 
        wei = self.Va(self.dropout(a)).squeeze()
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        out = wei@v
        return out
        
class Model(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_variables, N, sinusoidal = False):
        super().__init__()
        self.encoder = Encoder(d_model, n_heads, d_ff, num_variables, N, sinusoidal)
        self.fsa = FusionSelfAttention(d_model)
        self.proj = nn.Linear(d_model, 25)
    
    def forward(self, x, mask):
        out = self.encoder(x, mask)
        out = self.fsa(out, mask)
        out = out.masked_fill(mask.transpose(-2,-1)==0, 0)
        out = out.sum(dim = 1)
        out = self.proj(out)
        return out