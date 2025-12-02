import torch
import torch.nn as nn
from torch.nn import functional as F
import math

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from random_besov import FastWaveletBatchGenerator


###### Need to compile wavelet_fast.cu first:
# For linux, run "nvcc -arch=sm_80 -O3 -o wavelet_fast.so -shared -Xcompiler -fPIC,-O3 wavelet_fast.cu"
# For windows, run "nvcc -O3 wavelet_fast.cu -o wavelet_fast.dll -lcurand -shared"


###### Generate pretraining data for in-context regression transformer model

def generate_pretrain_data(X_list, params, d_embd, sigma):
    """
    Generates function data and prepares it for an in-context regression transformer model.

    Args:
        X_list (list of np.ndarray): List of input point arrays. Each array
            can have N points and dimension d.
        params (dict): A dictionary that contains the
            parameters for the functions: {'d', 'alpha', 'J', 'C0'}.
        d_embd (int): The embedding dimension for the transformer input.
        sigma (float): The standard deviation of Gaussian noise to add to the Y values.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - A tensor of shape (batch_size, N, d_embd) containing the
          concatenated and padded X and Y sequences. The last Y value
          in each sequence is masked with 0.
        - A tensor of shape (batch_size,) containing the true last Y value
          for each sequence, which serves as the regression target.
    """
    generator = FastWaveletBatchGenerator(wavelet='db6', cascade_level=8)
    Y_list = generator.generate_batch(X_list, params)
    generator.cleanup()

    device = torch.device('cuda')

    # Ensure d is correctly inferred even for 1D inputs
    if X_list[0].ndim == 1:
        d = 1
    else:
        d = X_list[0].shape[1]

    if (d + 3) > d_embd:
        raise ValueError(f"d_embd ({d_embd}) must be greater than or equal to the input dimension + 3 ({d + 3}).")

    batch_size = len(X_list)
    N = X_list[0].shape[0] 

    # Initialize output tensors
    output_sequences = torch.zeros((batch_size, N, d_embd), dtype=torch.float32, device=device)
    target_values = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for i in range(batch_size):
        # Get original data
        X_i = X_list[i]
        Y_i = Y_list[i]
        
        # Add noise
        if sigma > 0:
            Y_i += np.random.normal(scale=sigma, size=Y_i.shape)

        # Get the target value (the last Y value)
        target_values[i] = torch.as_tensor(Y_i[-1], device=device)
        
        # Prepare the input sequence
        # Convert X to a tensor
        X_i_torch = torch.from_numpy(np.asarray(X_i, dtype=np.float32)).to(device)
        if X_i_torch.ndim == 1:
            X_i_torch = X_i_torch.unsqueeze(-1)
        
        # Mask the last Y value
        Y_i[-1] = 0
        Y_i_torch = torch.from_numpy(np.asarray(Y_i, dtype=np.float32)).to(device)

        # Concatenate X and Y_masked
        # Y_masked needs to be shaped as (N, 1)
        xy_concat = torch.cat((X_i_torch, Y_i_torch.unsqueeze(-1)), dim=1)
        
        # Place the concatenated sequence into the padded output tensor
        output_sequences[i, :N, :(d + 1)] = xy_concat
        output_sequences[i, N - 1, d + 1] = 1

    return output_sequences, target_values



###### Transformer Model Definition
class ModelConfig:
    """Holds model hyperparameters."""
    def __init__(self, d_embd=1024, d_ffn=4096, n_layer=24, n_head=16, max_length=None):
        self.d_embd = d_embd
        self.d_ffn = d_ffn
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_length = max_length if max_length is not None else 1024

class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head attention layer with a projection at the end.
    This implementation is an encoder-style attention (bi-directional).
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_embd, 3 * config.d_embd)
        # Output projection
        self.c_proj = nn.Linear(config.d_embd, config.d_embd)
        self.n_head = config.n_head
        self.d_embd = config.d_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # NOTE: No causal mask is applied, making attention bi-directional.
        
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side-by-side

        # Output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ A single transformer block. """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_embd, config.d_ffn),
            nn.GELU(),
            nn.Linear(config.d_ffn, config.d_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class RegressionTransformer(nn.Module):
    """ A transformer encoder for regression tasks on sequences of vectors. """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

    def forward(self, x):
        """
        x: A float tensor of shape (batch_size, sequence_length, d_embd)
        """
        assert x.dim() == 3 and x.size(2) == self.config.d_embd, "Input tensor shape must be (batch, seq_len, d_embd)"
        b, t, c = x.size()
        assert t <= self.config.max_length, f"Cannot forward sequence of length {t}, max_length is only {self.config.max_length}"
        
        # Forward through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Return the last coordinate of the last token
        return x[:, -1, -1]


