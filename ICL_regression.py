import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import csv
import random

import numpy as np
# import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

from random_besov import FastWaveletBatchGenerator

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



###### Need to compile wavelet_fast.cu first:
# For linux, run "nvcc -arch=sm_80 -O3 -o wavelet_fast.so -shared -Xcompiler -fPIC,-O3 wavelet_fast.cu"
# For windows, run "nvcc -O3 wavelet_fast.cu -o wavelet_fast.dll -lcurand -shared"


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



###### Generate pretraining data for in-context regression transformer model
def generate_pretrain_data(batch_size, n, d, p, alpha, J, cascade_level, d_embd, sigma, C0=1.0):
    """
    Generates function data and prepares it for an in-context regression transformer model.

    Args:
        batch_size (int): number of functions in the batch.
        n (int): number of data points per function.
        d (int): input dimension.
        p (int): projection dimension.
        alpha (float): smoothness parameter.
        J (int): maximum wavelet scale.
        cascade_level (int): cascade level for wavelet generation.
        d_embd (int): The embedding dimension for the transformer input.
        sigma (float): The standard deviation of Gaussian noise to add to the Y values.
        C0 (float): scaling constant for the wavelet function.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - A tensor of shape (batch_size, N, d_embd) containing the
          concatenated and padded X and Y prompts. The last Y value
          in each sequence is masked with 0.
        - A tensor of shape (batch_size,) containing the true last Y value
          for each sequence, which serves as the regression target.
    """
    X_list = []
    X_projected_list = []
    for _ in range(batch_size):
        X = sample_unit_ball(n, d)
        X_list.append(X)
        X_projected_list.append(random_project(X, p))


    params = {'d': p, 'alpha': alpha, 'J': J, 'C0': C0}
    generator = FastWaveletBatchGenerator(wavelet='db6', cascade_level=cascade_level)
    Y_list = generator.generate_batch(X_projected_list, params)
    generator.cleanup()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (d + 3) > d_embd:
        raise ValueError(f"d_embd ({d_embd}) must be greater than or equal to the input dimension + 3 ({d + 3}).")

    batch_size = len(X_list)

    # Initialize output tensors
    prompts = torch.zeros((batch_size, n, d_embd), dtype=torch.float32, device=device)
    target_values = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for i in range(batch_size):
        # Get original data
        X_i = X_list[i]
        Y_i = Y_list[i]
        
        # Add noise
        Y_i += np.random.normal(scale=sigma, size=Y_i.shape)

        # Get the target value (the last Y value)
        target_values[i] = torch.as_tensor(Y_i[-1], dtype=torch.float32, device=device)
        
        # Prepare the input sequence
        # Convert X to a tensor
        X_i_torch = torch.from_numpy(np.asarray(X_i, dtype=np.float32)).to(device)
        if X_i_torch.ndim == 1:
            X_i_torch = X_i_torch.unsqueeze(-1)
        
        # Mask the last Y value
        Y_i[-1] = 0
        Y_i_torch = torch.from_numpy(np.asarray(Y_i, dtype=np.float32)).to(device)

        # Concatenate X and Y_masked
        # Y_masked needs to be shaped as (n, 1)
        xy_concat = torch.cat((X_i_torch, Y_i_torch.unsqueeze(-1)), dim=1)
        
        # Place the concatenated sequence into the padded output tensor
        prompts[i, :n, :(d + 1)] = xy_concat
        prompts[i, n - 1, d + 1] = 1

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
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.output_layer = nn.Linear(config.d_embd, 1)

    def forward(self, x):
        assert x.dim() == 3 and x.size(2) == self.config.d_embd
        b, t, c = x.size()
        assert t <= self.config.max_length

        for block in self.transformer.h:
            x = block(x)

        # use last token embedding for regression
        last_token = x[:, -1, :]           # (B, d_embd)
        y = self.output_layer(last_token)      # (B, 1)
        return y



###### Training Loop
batch_size = 512
n = 51
d = 10
sigma = 0.02

lr = 1e-4
weight_decay = 1e-2
num_epochs = 100000
checkpoint_freq = 200

# Model Config
d_embd=1024
d_ffn=4096
n_layer=24
n_head=16
config = ModelConfig(d_embd=d_embd, d_ffn=d_ffn, n_layer=n_layer, n_head=n_head)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model, Optimizer, Loss
model = RegressionTransformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

def _byte_state(state):
    """Convert checkpoint RNG state to a CPU uint8 tensor for torch.set_rng_state."""
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")

def save_checkpoint(epoch):
    """Persist model/optimizer and RNG state so training can be resumed."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(config),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "random_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all() 
    torch.save(checkpoint, Path("/rds/user/tm681/hpc-work/Adaptive_ICL/checkpoints") / f"epoch_{epoch+1:06d}.pt")
    torch.save(checkpoint, "/rds/user/tm681/hpc-work/Adaptive_ICL/latest.pt")


resume_path = Path("/rds/user/tm681/hpc-work/Adaptive_ICL/latest.pt")
if resume_path is not None and resume_path.exists():
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    torch.set_rng_state(_byte_state(ckpt["torch_rng_state"]))
    np.random.set_state(ckpt["numpy_rng_state"])
    if "random_rng_state" in ckpt:
        random.setstate(ckpt["random_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
        cuda_states = ckpt["cuda_rng_state"]
        if isinstance(cuda_states, (list, tuple)):
            cuda_states = [_byte_state(s) for s in cuda_states]
        else:
            cuda_states = [_byte_state(cuda_states)]
        torch.cuda.set_rng_state_all(cuda_states)
else:
    start_epoch = 0



# Training loop
warmup_steps = 5000
prompts, targets = None, None
for epoch in range(start_epoch, num_epochs):
    # Generate data for 5 epochs
    epoch_mod_5 = epoch % 5
    if epoch_mod_5 == 0:
        if prompts is not None:
            del prompts, targets
            torch.cuda.empty_cache()
        if epoch + 1 <= warmup_steps:
            prompts, targets = generate_pretrain_data(batch_size * 6, n=n, d=10, p=1, alpha=3, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
        elif epoch + 1 <= 3 * warmup_steps:
            batch_size_1 = int(max(1 - (epoch + 1 - warmup_steps) / (2 * warmup_steps), 0.25) * batch_size * 6)
            batch_size_2 = batch_size * 6 - batch_size_1
            prompts_1, targets_1 = generate_pretrain_data(batch_size=batch_size_1, n=n, d=10, p=1, alpha=3, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts_2, targets_2 = generate_pretrain_data(batch_size=batch_size_2, n=n, d=10, p=1, alpha=2, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts = torch.cat((prompts_1, prompts_2), dim=0)
            targets = torch.cat((targets_1, targets_2), dim=0)
            perm = torch.randperm(batch_size * 6, device=device)
            prompts = prompts[perm]
            targets = targets[perm] 
            del prompts_1, prompts_2
            del targets_1, targets_2
        elif epoch + 1 <= 5 * warmup_steps:
            batch_size_1 = int(0.25 * batch_size * 6)
            batch_size_2 = int(max(0.75 - (epoch + 1 - 3 * warmup_steps) / (2 * warmup_steps), 0.25) * batch_size * 6)
            batch_size_3 = batch_size * 6 - batch_size_1 - batch_size_2
            prompts_1, targets_1 = generate_pretrain_data(batch_size=batch_size_1, n=n, d=10, p=1, alpha=3, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts_2, targets_2 = generate_pretrain_data(batch_size=batch_size_2, n=n, d=10, p=1, alpha=2, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts_3, targets_3 = generate_pretrain_data(batch_size=batch_size_3, n=n, d=10, p=2, alpha=3, J=6, cascade_level=6, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts = torch.cat((prompts_1, prompts_2, prompts_3), dim=0)
            targets = torch.cat((targets_1, targets_2, targets_3), dim=0)
            perm = torch.randperm(batch_size * 6, device=device)
            prompts = prompts[perm]
            targets = targets[perm] 
            del prompts_1, prompts_2, prompts_3
            del targets_1, targets_2, targets_3
        else:
            batch_size_1 = int(0.25 * batch_size * 6)
            batch_size_2 = int(0.25 * batch_size * 6)
            batch_size_3 = int(max(0.5 - (epoch + 1 - 5 * warmup_steps) / (2 * warmup_steps), 0.25) * batch_size * 6)
            batch_size_4 = batch_size * 6 - batch_size_1 - batch_size_2 - batch_size_3
            prompts_1, targets_1 = generate_pretrain_data(batch_size=batch_size_1, n=n, d=10, p=1, alpha=3, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts_2, targets_2 = generate_pretrain_data(batch_size=batch_size_2, n=n, d=10, p=1, alpha=2, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts_3, targets_3 = generate_pretrain_data(batch_size=batch_size_3, n=n, d=10, p=2, alpha=3, J=6, cascade_level=6, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts_4, targets_4 = generate_pretrain_data(batch_size=batch_size_4, n=n, d=10, p=1, alpha=1, J=10, cascade_level=8, d_embd=d_embd, sigma=sigma, C0=1.0)
            prompts = torch.cat((prompts_1, prompts_2, prompts_3, prompts_4), dim=0)
            targets = torch.cat((targets_1, targets_2, targets_3, targets_4), dim=0)
            perm = torch.randperm(batch_size * 6, device=device)
            prompts = prompts[perm]
            targets = targets[perm] 
            del prompts_1, prompts_2, prompts_3, prompts_4
            del targets_1, targets_2, targets_3, targets_4

    # Forward pass
    outputs = model(prompts[epoch_mod_5*batch_size : (epoch_mod_5+1)*batch_size, :, :])
    train_loss = criterion(outputs.squeeze(), targets[epoch_mod_5*batch_size : (epoch_mod_5+1)*batch_size])
    
    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            outputs = model(prompts[5*batch_size : 6*batch_size, :, :])
            val_loss = criterion(outputs.squeeze(), targets[5*batch_size : 6*batch_size])
        model.train()
        if epoch + 1 == 5:
            with open("/rds/user/tm681/hpc-work/Adaptive_ICL/losses.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([train_loss.item(), val_loss.item()])
        else:
            with open("/rds/user/tm681/hpc-work/Adaptive_ICL/losses.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([train_loss.item(), val_loss.item()])        

    if (epoch + 1) % checkpoint_freq == 0:
        save_checkpoint(epoch)


