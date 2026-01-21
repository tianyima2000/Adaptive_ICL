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
# import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

from random_besov import FastWaveletBatchGenerator

SEED = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CHECKPOINT_ROOT = Path("/rds/user/tm681/hpc-work/Adaptive_ICL_multi_GPU")
CHECKPOINT_DIR = CHECKPOINT_ROOT / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_ROOT / "latest.pt"
LOSSES_CSV = CHECKPOINT_ROOT / "losses.csv"



###### Need to compile wavelet_fast.cu first:
# For linux, run "nvcc -arch=sm_80 -O3 -o wavelet_fast.so -shared -Xcompiler -fPIC,-O3 wavelet_fast.cu"
# For windows, run "nvcc -O3 wavelet_fast.cu -o wavelet_fast.dll -lcurand -shared"


def set_seed(seed, rank=0):
    """Seed python, numpy, and torch (including CUDA) deterministically per-rank."""
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def setup_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    """Return the underlying model when wrapped with DistributedDataParallel."""
    return model.module if isinstance(model, DDP) else model


def distributed_mean(value, world_size):
    """Average a scalar tensor across processes if distributed is initialized."""
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


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
def generate_pretrain_data(batch_size, n, d, p, alpha, J, cascade_level, d_max, sigma, device=None, C0=1.0):
    
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
        
        # Add noise
        Y_i += np.random.normal(scale=sigma, size=Y_i.shape)

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


        
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight        
        


    
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



class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # because last_epoch starts at -1

        if step <= self.warmup_steps:
            # Linear warmup: lr increases from 0 → base_lr
            return [
                base_lr * step / float(self.warmup_steps)
                for base_lr in self.base_lrs
            ]

        # After warmup: cosine decay
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return [base_lr * cosine_decay for base_lr in self.base_lrs]




###### Training Loop
batch_size = 50  # Per-GPU batch size when running with torchrun
n = 31
d = 5
sigma = 0.01
C0 = 2.0

lr = 3e-4
weight_decay = 1e-3
num_epochs = 200000
checkpoint_freq = 5000
data_gen_freq = 20  # Generate new data every data_gen_freq epochs

# Model Config
d_embd = 1024
d_ffn = 4096
n_layer = 24
n_head = 16
config = ModelConfig(d_embd=d_embd, d_ffn=d_ffn, n_layer=n_layer, n_head=n_head)

criterion = nn.MSELoss()


def _byte_state(state):
    """Convert checkpoint RNG state to a CPU uint8 tensor for torch.set_rng_state."""
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu")


def save_checkpoint(epoch, model, optimizer, rank):
    """Persist model/optimizer and RNG state so training can be resumed."""
    if not is_main_process(rank):
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(config),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "random_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, CHECKPOINT_DIR / f"epoch_{epoch+1:06d}.pt")
    torch.save(checkpoint, LATEST_CHECKPOINT)


def load_checkpoint(model, optimizer, device, rank):
    """Load checkpoint state across all ranks if it exists, then reseed per-rank."""
    start_epoch = 0
    if LATEST_CHECKPOINT.exists():
        ckpt = torch.load(LATEST_CHECKPOINT, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

        if rank == 0:
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

    if dist.is_available() and dist.is_initialized():
        base_seed = torch.tensor(SEED if rank == 0 else 0, device=device)
        dist.broadcast(base_seed, src=0)
        set_seed(int(base_seed.item()), rank)
        dist.barrier()
    else:
        set_seed(SEED, rank)

    return start_epoch


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(SEED, rank)

    base_model = RegressionTransformer(config, d+2).to(device)
    if world_size > 1:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True)
    else:
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = load_checkpoint(model, optimizer, device, rank)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=2000, total_steps=num_epochs, last_epoch=start_epoch-1)
    model.train()

    warmup_steps = 10000
    prompts, targets = None, None
    for epoch in range(start_epoch, num_epochs):
        # Generate data for data_gen_freq epochs
        epoch_mod_data_gen_freq = epoch % data_gen_freq
        if epoch_mod_data_gen_freq == 0:
            if prompts is not None:
                del prompts, targets
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if epoch + 1 <= 1000:
                prompts, targets = generate_pretrain_data(batch_size * (data_gen_freq + 1), n=n, d=1, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
            elif epoch + 1 <= 2000:
                prompts, targets = generate_pretrain_data(batch_size * (data_gen_freq + 1), n=n, d=2, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
            elif epoch + 1 <= 3000:
                prompts, targets = generate_pretrain_data(batch_size * (data_gen_freq + 1), n=n, d=3, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
            elif epoch + 1 <= 4000:
                prompts, targets = generate_pretrain_data(batch_size * (data_gen_freq + 1), n=n, d=4, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
            elif epoch + 1 <= warmup_steps:
                prompts, targets = generate_pretrain_data(batch_size * (data_gen_freq + 1), n=n, d=d, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
            elif epoch + 1 <= 3 * warmup_steps:
                batch_size_1 = int(max(1 - (epoch + 1 - warmup_steps) / (2 * warmup_steps), 0.25) * batch_size * (data_gen_freq + 1))
                batch_size_2 = batch_size * (data_gen_freq + 1) - batch_size_1
                alpha_2 = max(4 - (epoch + 1 - warmup_steps) / warmup_steps, 3)
                prompts_1, targets_1 = generate_pretrain_data(batch_size=batch_size_1, n=n, d=d, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts_2, targets_2 = generate_pretrain_data(batch_size=batch_size_2, n=n, d=d, p=1, alpha=alpha_2, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts = torch.cat((prompts_1, prompts_2), dim=0)
                targets = torch.cat((targets_1, targets_2), dim=0)
                perm = torch.randperm(batch_size * (data_gen_freq + 1), device=device)
                prompts = prompts[perm]
                targets = targets[perm]
                del prompts_1, prompts_2
                del targets_1, targets_2
            elif epoch + 1 <= 5 * warmup_steps:
                batch_size_1 = int(0.25 * batch_size * (data_gen_freq + 1))
                batch_size_2 = int(max(0.75 - (epoch + 1 - 3 * warmup_steps) / (2 * warmup_steps), 0.25) * batch_size * (data_gen_freq + 1))
                batch_size_3 = batch_size * (data_gen_freq + 1) - batch_size_1 - batch_size_2
                alpha_3 = max(3 - 0.5 * (epoch + 1 - 3 * warmup_steps) / warmup_steps, 2.5)
                prompts_1, targets_1 = generate_pretrain_data(batch_size=batch_size_1, n=n, d=d, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts_2, targets_2 = generate_pretrain_data(batch_size=batch_size_2, n=n, d=d, p=1, alpha=3, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts_3, targets_3 = generate_pretrain_data(batch_size=batch_size_3, n=n, d=d, p=1, alpha=alpha_3, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts = torch.cat((prompts_1, prompts_2, prompts_3), dim=0)
                targets = torch.cat((targets_1, targets_2, targets_3), dim=0)
                perm = torch.randperm(batch_size * (data_gen_freq + 1), device=device)
                prompts = prompts[perm]
                targets = targets[perm]
                del prompts_1, prompts_2, prompts_3
                del targets_1, targets_2, targets_3
            else:
                batch_size_1 = int(0.25 * batch_size * (data_gen_freq + 1))
                batch_size_2 = int(0.25 * batch_size * (data_gen_freq + 1))
                batch_size_3 = int(max(0.5 - (epoch + 1 - 5 * warmup_steps) / (2 * warmup_steps), 0.25) * batch_size * (data_gen_freq + 1))
                batch_size_4 = batch_size * (data_gen_freq + 1) - batch_size_1 - batch_size_2 - batch_size_3
                alpha_4 = max(3 - (epoch + 1 - 5 * warmup_steps) / warmup_steps, 2)
                prompts_1, targets_1 = generate_pretrain_data(batch_size=batch_size_1, n=n, d=d, p=1, alpha=4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts_2, targets_2 = generate_pretrain_data(batch_size=batch_size_2, n=n, d=d, p=1, alpha=3, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts_3, targets_3 = generate_pretrain_data(batch_size=batch_size_3, n=n, d=d, p=1, alpha=2.5, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts_4, targets_4 = generate_pretrain_data(batch_size=batch_size_4, n=n, d=d, p=1, alpha=alpha_4, J=10, cascade_level=8, d_max=d, sigma=sigma, device=device, C0=C0)
                prompts = torch.cat((prompts_1, prompts_2, prompts_3, prompts_4), dim=0)
                targets = torch.cat((targets_1, targets_2, targets_3, targets_4), dim=0)
                perm = torch.randperm(batch_size * (data_gen_freq + 1), device=device)
                prompts = prompts[perm]
                targets = targets[perm]
                del prompts_1, prompts_2, prompts_3, prompts_4
                del targets_1, targets_2, targets_3, targets_4

        # Forward pass
        batch_prompts = prompts[epoch_mod_data_gen_freq * batch_size : (epoch_mod_data_gen_freq + 1) * batch_size, :, :]
        batch_targets = targets[epoch_mod_data_gen_freq * batch_size : (epoch_mod_data_gen_freq + 1) * batch_size]
        outputs = model(batch_prompts)
        train_loss = criterion(outputs.squeeze(), batch_targets)
        reduced_train_loss = distributed_mean(train_loss.detach().clone(), world_size)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % data_gen_freq == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(prompts[data_gen_freq * batch_size : (data_gen_freq + 1) * batch_size, :, :])
                val_loss = criterion(outputs.squeeze(), targets[data_gen_freq * batch_size : (data_gen_freq + 1) * batch_size])
                reduced_val_loss = distributed_mean(val_loss.detach().clone(), world_size)
            model.train()

            if is_main_process(rank):
                LOSSES_CSV.parent.mkdir(parents=True, exist_ok=True)
                if epoch + 1 == data_gen_freq:
                    with open(LOSSES_CSV, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([reduced_train_loss.item(), reduced_val_loss.item()])
                else:
                    with open(LOSSES_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([reduced_train_loss.item(), reduced_val_loss.item()])

        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(epoch, model, optimizer, rank)

    cleanup_distributed()


if __name__ == "__main__":
    main()