import ctypes
import numpy as np
import pywt
import os
import shutil
import subprocess
import sys

MAX_GRID_SIZE = 4096
DEFAULT_CUDA_ARCH = os.environ.get("WAVELET_FAST_ARCH", "sm_70")


def _library_basename():
    if sys.platform.startswith("win"):
        return "wavelet_fast.dll"
    if sys.platform == "darwin":
        return "libwavelet_fast.dylib"
    return "wavelet_fast.so"


def _build_cuda_library(lib_path, cu_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError(
            "nvcc not found in PATH. Load the CUDA toolkit module or install CUDA before building."
        )
    os.makedirs(os.path.dirname(lib_path), exist_ok=True)
    cmd = [
        nvcc,
        "-O3",
        f"-arch={DEFAULT_CUDA_ARCH}",
        cu_path,
        "-o",
        lib_path,
        "-lcurand",
    ]
    if sys.platform.startswith("win"):
        cmd.extend(["-shared"])
    else:
        cmd.extend(["-shared", "-Xcompiler", "-fPIC"])
    subprocess.run(cmd, check=True)


def _ensure_cuda_library():
    lib_dir = os.path.dirname(__file__)
    lib_name = _library_basename()
    lib_path = os.path.join(lib_dir, lib_name)
    if os.path.exists(lib_path):
        return lib_path
    cu_path = os.path.join(lib_dir, "wavelet_fast.cu")
    if not os.path.exists(cu_path):
        raise FileNotFoundError(
            f"Missing CUDA source {cu_path}; cannot build {lib_name} automatically."
        )
    _build_cuda_library(lib_path, cu_path)
    return lib_path

class FastWaveletCUDA:
    """
    Ultra-fast CUDA wavelet evaluator with precomputed coefficients.
    
    Usage:
        rwb = FastWaveletCUDA(d=2, alpha=0.5, J=3, seed=42)
        X = np.random.rand(1000000, 2).astype(np.float32)
        y = rwb(X)
    """
    
    def __init__(self, d, alpha, J, seed, C0=1.0, wavelet='db6', cascade_level=8):
        """
        Initialize the fast wavelet evaluator.
        
        Parameters:
        -----------
        d : int
            Dimension (1, 2, 3, or 4)
        alpha : float
            Besov smoothness parameter
        J : int
            Maximum wavelet scale
        seed : int
            Random seed for coefficients
        C0 : float
            Overall amplitude
        wavelet : str
            Wavelet family (default: 'db6')
        cascade_level : int
            Wavelet function accuracy
        """
        self.d = d
        self.alpha = alpha
        self.J = J
        self.C0 = C0
        self.seed = seed
        
        # Get wavelet data from PyWavelets
        w = pywt.Wavelet(wavelet)
        phi, psi, grid = w.wavefun(level=cascade_level)
        
        # Convert to float32 for CUDA (ensure contiguous)
        self.phi = np.ascontiguousarray(phi.astype(np.float32))
        self.psi = np.ascontiguousarray(psi.astype(np.float32))
        self.grid = np.ascontiguousarray(grid.astype(np.float32))

        if self.grid.size > MAX_GRID_SIZE:
            raise ValueError(
                f"Wavelet grid has {self.grid.size} points but CUDA kernel was "
                f"built with MAX_GRID_SIZE={MAX_GRID_SIZE}. Reduce the cascade "
                "level or rebuild wavelet_fast.cu with a larger constant."
            )
        
        # Load or build the CUDA library on demand
        lib_path = _ensure_cuda_library()
        self.lib = ctypes.CDLL(lib_path)
        
        # Setup function signature
        self.lib.evaluate_wavelet_fast.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # X
            ctypes.POINTER(ctypes.c_float),  # y
            ctypes.c_int,                     # N
            ctypes.c_int,                     # d
            ctypes.POINTER(ctypes.c_float),  # phi
            ctypes.POINTER(ctypes.c_float),  # psi
            ctypes.POINTER(ctypes.c_float),  # grid
            ctypes.c_int,                     # grid_len
            ctypes.c_float,                   # alpha
            ctypes.c_int,                     # J
            ctypes.c_float,                   # C0
            ctypes.c_uint32                   # seed
        ]
        self.lib.evaluate_wavelet_fast.restype = None
        
        print(f"FastWaveletCUDA initialized: d={d}, alpha={alpha}, J={J}, seed={seed}")
        print(f"Wavelet grid size: {len(self.grid)}")
    
    def __call__(self, X):
        """
        Evaluate the wavelet function at points X.
        
        Parameters:
        -----------
        X : ndarray, shape (N, d)
            Points to evaluate, with coordinates in [0, 1]
            Will be converted to float32 if necessary
        
        Returns:
        --------
        y : ndarray, shape (N,)
            Function values at the points
        """
        # Ensure correct format
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X.ndim == 1:
            X = X.reshape(-1, self.d)
        
        if X.shape[1] != self.d:
            raise ValueError(f"X must have shape (N, {self.d}), got {X.shape}")
        
        N = X.shape[0]
        y = np.ascontiguousarray(np.zeros(N, dtype=np.float32))
        
        # Call CUDA function
        self.lib.evaluate_wavelet_fast(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N,
            self.d,
            self.phi.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.psi.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(self.grid),
            self.alpha,
            self.J,
            self.C0,
            self.seed
        )
        
        return y
    
    def evaluate_batch(self, X, batch_size=100000):
        """
        Evaluate in batches to manage memory for very large datasets.
        
        Parameters:
        -----------
        X : ndarray, shape (N, d)
            Points to evaluate
        batch_size : int
            Number of points per batch
        
        Returns:
        --------
        y : ndarray, shape (N,)
            Function values
        """
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        N = X.shape[0]
        y = np.zeros(N, dtype=np.float32)
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            y[i:end] = self(X[i:end])
        
        return y



rwb = FastWaveletCUDA(d=1, alpha=1, J=10, seed=42)
X = np.linspace(0,1,10**12)
y = rwb(X)
np.savez("output.npz", X=X, y=y)
    


