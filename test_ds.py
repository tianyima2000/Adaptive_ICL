import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import csv
import random

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

from random_besov_ds import FastWaveletBatchGenerator_ds

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CHECKPOINT_ROOT = Path("/rds/user/tm681/hpc-work/Adaptive_ICL_multi_GPU")
LATEST_CHECKPOINT = CHECKPOINT_ROOT / "latest.pt"
TEST_CSV = CHECKPOINT_ROOT / "test_results_ds.csv"


###### Need to compile wavelet_fast.cu first:
# For linux, run "nvcc -arch=sm_80 -O3 -o wavelet_fast_ds.so -shared -Xcompiler -fPIC,-O3 wavelet_fast_ds.cu"
# For windows, run "nvcc -O3 wavelet_fast_ds.cu -o wavelet_fast_ds.dll -lcurand -shared"


###### Sampling Utilities
def sample_unit_ball(n, d):
    z = np.random.normal(size=(n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    r = np.random.uniform(size=(n, 1)) ** (1 / d)
    return r * z


def sample_stiefel(d, p):
    if p > d:
        raise ValueError("p must be <= d")

    Z = np.random.normal(size=(d, p))
    Q, R = np.linalg.qr(Z)

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q *= signs
    return Q


def random_project(X, p):
    d = X.shape[1]
    U = sample_stiefel(d, p)
    X_projected = (np.matmul(X, U) + 1) / 2
    return X_projected



###### Generate test data for in-context regression transformer model
def generate_test_data(batch_size, n, d, p, alpha, J, cascade_level, d_max, sigma, device=None, C0=1.0):
    
    X_list = []
    X_projected_list = []
    for _ in range(batch_size):
        X = sample_unit_ball(n, d)
        X_list.append(X)
        X_projected_list.append(random_project(X, p))


    params = {'d': p, 'alpha': alpha, 'J': J, 'C0': C0}
    generator = FastWaveletBatchGenerator_ds(wavelet='db6', cascade_level=cascade_level)
    Y_list = generator.generate_batch(X_projected_list, params)
    generator.cleanup()
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = len(X_list)

    # Initialize output tensors with max dimension
    # Shape: (Batch, N, d_max + 2)
    # [ 0...d_max-1: Features ] [ d_max: Y values ] [ d_max+1: Mask ]
    prompts = torch.zeros((batch_size, n, d_max + 2), dtype=torch.float32, device=device)
    target_values = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for i in range(batch_size):
        X_i = X_list[i]
        Y_i = Y_list[i]
        
        # Add noise to prompt
        Y_i[0:(n-1)] += np.random.normal(scale=sigma, size=Y_i[0:(n-1)].shape)

        # Get the target value
        target_values[i] = torch.as_tensor(Y_i[-1], dtype=torch.float32, device=device)
        
        # Prepare inputs
        X_i_torch = torch.from_numpy(np.asarray(X_i, dtype=np.float32)).to(device)
        if X_i_torch.ndim == 1:
            X_i_torch = X_i_torch.unsqueeze(-1)
        
        # Mask the last Y value for the prompt
        Y_i[-1] = 0
        Y_i_torch = torch.from_numpy(np.asarray(Y_i, dtype=np.float32)).to(device)
        
        # 1. Place X in the first d columns
        prompts[i, :n, :d] = X_i_torch
        
        # 2. Place Y in the FIXED column (d_max)
        prompts[i, :n, d_max] = Y_i_torch
        
        # 3. Place PE in the last column (d_max + 1)
        prompts[i, n - 1, d_max + 1] = 1.0

    return prompts, target_values
        
        


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
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(input_dim, config.d_embd)
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.output_layer = nn.Linear(config.d_embd, 1)

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.transformer.h:
            x = block(x)

        # use last token embedding for regression
        last_token = x[:, -1, :]           # (B, d_embd)
        y = self.output_layer(last_token)      # (B, 1)
        return y
    

n = 31
d = 5
sigma = 0.01
C0 = 2.0

# Model Config
d_embd = 1024
d_ffn = 4096
n_layer = 24
n_head = 16
config = ModelConfig(d_embd=d_embd, d_ffn=d_ffn, n_layer=n_layer, n_head=n_head)

criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionTransformer(config, d+2).to(device)
ckpt = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)
def unwrap_model(model):
    """Return the underlying model when wrapped with DistributedDataParallel."""
    return model.module if isinstance(model, DDP) else model
unwrap_model(model).load_state_dict(ckpt["model_state_dict"])

def set_seed(seed):
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2025)


loss_1, loss_2, loss_3, loss_4 = [], [], [], []
model.eval()
with torch.no_grad():
    for i in range(100):
        prompts, targets = generate_test_data(batch_size = 100, n=n, d=d, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
        outputs = model(prompts)
        loss_1.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)

        prompts, targets = generate_test_data(batch_size = 100, n=n, d=d, p=1, alpha=3, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
        outputs = model(prompts)
        loss_2.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)

        prompts, targets = generate_test_data(batch_size = 100, n=n, d=d, p=1, alpha=2.5, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
        outputs = model(prompts)
        loss_3.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)

        prompts, targets = generate_test_data(batch_size = 100, n=n, d=d, p=1, alpha=2, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
        outputs = model(prompts)
        loss_4.append((outputs.cpu().detach().numpy().reshape(-1) - targets.cpu().detach().numpy()) ** 2)

loss_1 = np.concatenate(loss_1)
loss_2 = np.concatenate(loss_2)
loss_3 = np.concatenate(loss_3)
loss_4 = np.concatenate(loss_4)

df = pd.DataFrame({
    "alpha_4": loss_1,
    "alpha_3": loss_2,
    "alpha_2_5": loss_3,
    "alpha_2": loss_4
})

df.to_csv(TEST_CSV, index=False)