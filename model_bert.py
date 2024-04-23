import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import math

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau.unsqueeze(-1), self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau.unsqueeze(-1), self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class ContinuousValueEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(1, d_model*2)
        self.U = nn.Linear(d_model*2, d_model)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = self.W(x.unsqueeze(2))
        out = self.tanh(out)
        out = self.U(out)
        return out


class VariableEmbedding(nn.Module):
    def __init__(self, d_model, num_variables):
        super().__init__()
        self.embedding = nn.Embedding(num_variables+1, d_model)
        
    def forward(self, x):
        return self.embedding(x)
    
    

class Embedding(nn.Module):
    def __init__(self, d_model, num_variables, sinusoidal, pre_training = False):
        super().__init__()
        self.sinusoidal = sinusoidal
        self.cvs_value = ContinuousValueEmbedding(d_model)
        if sinusoidal:
            self.cvs_time = SineActivation(1, d_model)
        if sinusoidal == "both":
            self.cvs_time = ContinuousValueEmbedding(d_model)
            self.sin_time = SineActivation(1, d_model)
        else:
            self.cvs_time = ContinuousValueEmbedding(d_model)
        self.var_embed = VariableEmbedding(d_model, num_variables)
        self.d_model = d_model
        if pre_training==True:
            self.mask_embedding = nn.Embedding(1, d_model).to(DEVICE)
        self.class_embedding = nn.Embedding(1, d_model).to(DEVICE)
    def forward(self, encoder_input, pre_training_mask = None):
        time = encoder_input[0]
        variable = encoder_input[1]
        value = encoder_input[2]
        if self.sinusoidal == "both":
            time_embed = self.cvs_time(time) + self.sin_time(time)
        else:
            time_embed = self.cvs_time(time)
        if pre_training_mask is not None:
            #pretraining_mask = (Batch_size, MAX_LEN)
            #value = (Batch_size, MAX_LEN)
            pre_training_mask = pre_training_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
            #pretraining_mask = (Batch_size, MAX_LEN, d_model)
            mask_token = torch.tensor([[0]], dtype=torch.int64).to(DEVICE)
            mask_embed = self.mask_embedding(mask_token).to(DEVICE)
            #mask_embed = (1, d_model)
            mask_embed = mask_embed.expand(value.size(0), value.size(1), -1).to(DEVICE)
            #mask_embed = (Batch_size, MAX_LEN, d_model)
            value_embed = self.cvs_value(value)
            #value_embed = (Batch_size, MAX_LEN, d_model)
            embeds = torch.where(pre_training_mask==0, value_embed, mask_embed).to(DEVICE)
        else:
            embeds = self.cvs_value(value)
        embed = time_embed + embeds + self.var_embed(variable)
        B = embed.size(0)
        class_embed = self.class_embedding(torch.tensor([[0]], dtype=torch.int64).to(DEVICE))
        class_embed = class_embed.repeat(B, 1, 1)
        
        
        embed = torch.cat((embed, class_embed), dim=1)
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
        self.casual_mask = torch.tril(torch.ones((1, 1026 , 1026)), diagonal=0).type(torch.int).to(DEVICE)
        self.register_buffer("casual_mask_buffer", self.casual_mask)
    def forward(self,x, mask): 
        B, L, C = x.size()
        q = self.Q(x) 
        k = self.K(x)
        v = self.V(x) 
        weights = q@k.transpose(-2,-1)*k.shape[-1]**(-0.5) 
        weights = weights.masked_fill(self.casual_mask[:,:L,:L] == 0, float('-inf'))
        weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1) 
        self.dropout(weights)
        out = weights @ v
        return F.relu(out) 

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
    def __init__(self, d_model, n_heads, d_ff, num_variables , N, sinusoidal, pre_training = False):
        super().__init__()
        self.embedding = Embedding(d_model, num_variables, sinusoidal, pre_training)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(N)])
        self.N = N
    
    def forward(self, encoder_input, mask, pretrain_mask=None):
        time = encoder_input[0]
        variable = encoder_input[1]
        value = encoder_input[2]
        x = self.embedding((time, variable, value), pretrain_mask)
        for block in self.encoder_blocks:
            x = block(x, mask)
        return x

# class FusionSelfAttention(nn.Module):
#     def __init__(self, d_model, dropout = 0.2):
#         super().__init__()
#         self.Wa = nn.Linear(d_model, d_model)
#         self.Ua = nn.Linear(d_model, d_model)
#         self.Va = nn.Linear(d_model, 1)
#         self.dropout = nn.Dropout(0.2)
        
#     def forward(self, out, mask):
#         q = out.unsqueeze(2) 
#         k = out.unsqueeze(1) 
#         v = out 
#         a = F.tanh(self.Wa(q) + self.Ua(k)) 
#         wei = self.Va(self.dropout(a)).squeeze()
#         wei = wei.masked_fill(mask == 0, float('-inf'))
#         wei = F.softmax(wei, dim = -1)
#         wei = self.dropout(wei)
#         out = wei@v
#         return out

    
# class FusionSelfAttention(nn.Module):
#     def __init__(self, input_size, da = 64):
#         super(FusionSelfAttention, self).__init__()
#         self.input_size = input_size
#         self.output_size = 1
    
#         self.Wa = nn.Linear(input_size, da)
#         self.u = nn.Linear(da, 1, bias=False)
        
#     def forward(self, c, mask=None):
        
#         intermediate = torch.tanh(self.Wa(c))
#         scores = self.u(intermediate)
        
#         if mask is not None:
#             scores = scores.squeeze(-1).masked_fill(mask == 0, float('-inf'))
#         attention_weights = F.softmax(scores, dim=-1)
#         return attention_weights


class StratsModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_variables, N, sinusoidal = False, pre_training=False):
        super().__init__()
        self.encoder = Encoder(d_model, n_heads, d_ff, num_variables, N, sinusoidal, pre_training)
        # self.fsa = FusionSelfAttention(d_model)
    
    def forward(self, x, mask, pre_training_mask=None):
        
        out = self.encoder(x, mask, pre_training_mask)
        # weights = self.fsa(out, mask)
        # out = out.masked_fill(mask.transpose(-2,-1)==0, 0)
        # out = out.sum(dim = 1)
        #out = self.proj(out)
        return out#.squeeze(-1)

class ForecastModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x, mask = None):
        return self.fc(x).squeeze(-1)

class PredictionModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.fsa = FusionSelfAttention(d_model)
        self.proj = nn.Linear(d_model, 25)
    def forward(self, x, Mask):
        # weights = self.fsa(x)
        # out = torch.sum(weights * x, dim=1)
        # out = x.masked_fill(mask.transpose(-2, -1) == 0, 0)
        # out = out.sum(dim=1)
        out = self.proj(x[:, -1, :])
        return out
    

class Model(nn.Module):
    def __init__(self, strats_model, out_model):
        super().__init__()
        self.strats_model = strats_model
        self.out_model = out_model
    def forward(self, x, mask, pre_training_mask = None):
        out = self.strats_model(x, mask, pre_training_mask)
        out = self.out_model(out, mask)
        return out