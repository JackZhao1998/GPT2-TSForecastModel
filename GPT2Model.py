# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:28:08 2023

@author: jackz
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead

        assert (
            d_model % nhead == 0
        ), "Embedding size (d_model) needs to be divisible by number of heads"

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, query, key, value, d_k, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) /  np.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -np.inf)

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, value)
        return output

    def forward(self, query, key, value, mask=None):
        I = query.shape[0]

        # perform linear operation and split into N heads
        query = self.q_linear(query).view(I, -1, self.nhead, self.d_k)
        key = self.k_linear(key).view(I, -1, self.nhead, self.d_k)
        value = self.v_linear(value).view(I, -1, self.nhead, self.d_k)

        # transpose to get dimensions I * nhead * J * d_k
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # calculate attention
        scores = self.attention(query, key, value, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(I, -1, self.d_model)
        output = self.out(concat)

        return output
    
class Decoder_Layer(nn.Module):
    """
    Decoder Block

    Parameters
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    dropout:
        The dropout value
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.Attention = MultiHeadAttention(d_model, nhead)

        self.feedForward = nn.Sequential(
            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Linear(64,d_model),
            nn.Dropout(dropout)
            )

        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, mask):

        # Attention
        residual = q
        x = self.Attention(query=q, key=kv, value=kv, mask=mask)
        x = self.dropout(x)
        x = self.layerNorm1(x + residual)

        # Feed Forward
        residual = x
        x = self.feedForward(x)
        x = self.layerNorm2(x + residual)

        return x
    
def positional_encoding(batch_size, length, d_model, obs_time, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Positional Encoding for each visit

        Parameters
        ----------
        batch_size:
            Number of subjects in batch
        length:
            Number of visits
        d_model:
            Dimension of the model vector
        obs_time:
            Observed/recorded time of each visit
        """

        pe = torch.zeros(batch_size, length, d_model).to(device)
        _time = obs_time.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).reshape(1, 1, -1).to(device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        return pe

class Decoder(nn.Module):
    """
    Decoder Block

    Parameters
    ----------
    d_input:
        Number of input features
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_input,
                 d_model,
                 nhead,
                 num_decoder_layers,
                 dropout):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
            )

        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model,nhead,dropout)
                                             for _ in range(num_decoder_layers)])

    def forward(self, x, mask, obs_time):

        # Linear Embedding
        x = self.embedding(x)

        # Positional Embedding
        x = x + positional_encoding(
            x.shape[0], x.shape[1], x.shape[2], obs_time)

        # Decoder Layers
        for layer in self.decoder_layers:
            x = layer(x, x, mask)

        return x


class Decoder_p(nn.Module):
    """
    Decoder Block

    Parameters
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_model,
                 nhead,
                 num_decoder_layers,
                 dropout):
        super().__init__()

        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model,nhead,dropout)
                                             for _ in range(num_decoder_layers)])

    def forward(self, q, kv, mask, pred_time):
        # Positional Embedding
        q = q + positional_encoding(
            q.shape[0], q.shape[1], q.shape[2], pred_time)

        # Decoder Layers
        for layer in self.decoder_layers:
            output = layer(q, kv, mask)
            q = output
            kv = output

        return output

class GPT2Model(nn.Module):
    """
    An adaptation of the transformer model (Attention is All you Need)
    for survival analysis.

    Parameters
    ----------
    d_input:
        Dimension of inputs features
    d_model:
        Dimension of the input vector (post embedding) to hidden layers
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_input,
                 d_output,
                 d_model = 32,
                 nhead = 4,
                 num_decoder_layers = 3,
                 dropout = 0.2):
        super().__init__()

        self.decoder = Decoder(d_input, d_model, nhead, num_decoder_layers, dropout)

        self.decoder_pred = Decoder_p(d_model, nhead, num_decoder_layers, dropout)

        self.RELU = nn.ReLU()

        self.pred = nn.Sequential(
            nn.Linear(d_model, d_output)
            )


    def forward(self, x, mask, obs_time, pred_time):
        # Decoder Layers
        x = self.decoder(x, mask, obs_time) #hidden state from past time states

        # Decoder Layer with prediction time embedding
        x = self.decoder_pred(x, x, mask, pred_time) #hidden state for next time states

        # Output layer
        yhat = self.pred(x)

        return yhat
    
#define model configs
class GPT2ModelConfig:
    def __init__(self):
        #set seed
        self.seed = 0

        #model structure hyperparameter
        self.feature_list = ['amplitude0','freq0','phase0','trend','time']
        self.d_input = len(self.feature_list) #number of time invariant features
        self.target_var_list = ['noisy_signal']
        self.d_output = len(self.target_var_list) #number of targets to be predicted
        self.time = 'time'

        self.d_model = 128 #size of hidden layers in the model
        self.num_decoder_lay = 2 #number of decoder layers to be used in the model
        self.n_head = 8 #numbers of attention head for the model
        self.dropout = 0.2 #proportion of dropout features during training
         
        #Training Hyperparameters
        self.n_epoch = 50
        self.batch_size = 20

        #used for optimizer&scheduler
        self.learning_rate = 0
        self.betas = (0.9, 0.99)
        self.eps = 1e-9
        self.warmup_steps = 200
        self.factor = 0.2
        self.dropout = 0.2
        
        #path
        self.path = path+'model_name.pt'
