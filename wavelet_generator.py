import ctypes
import torch
import numpy as np
import os
import random
import platform
import subprocess
import pywt

def _compile_cuda_source(lib_path: str, source_path: str = "wavelet_fast.cu"):
    """Compiles the CUDA source file into a shared library."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"CUDA source file not found at {source_path}")

    print(f"Compiling {source_path} into {lib_path}...")
    
    command = [
        "nvcc",
        "-O3",
        "-arch=sm_70",
        "-shared",
        "-o", lib_path,
        source_path,
        "-Xcompiler", "-fPIC"
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Compilation successful.")
        if result.stdout:
            print("Compiler output:\n", result.stdout)
    except FileNotFoundError:
        raise RuntimeError("`nvcc` command not found. Is the CUDA Toolkit installed and in your PATH?")
    except subprocess.CalledProcessError as e:
        print("Compilation failed.")
        print("Error output:\n", e.stderr)
        raise RuntimeError("CUDA compilation failed.") from e

class WaveletGenerator:
    """
    A Python wrapper for the stream-aware wavelet_fast shared library.
    
    Handles auto-compilation, loading the library, and orchestrating parallel 
    generation of wavelet datasets using PyTorch and PyWavelets.
    """
    def __init__(self, grid: np.ndarray):
        """
        Initializes the generator, auto-compiles CUDA code, loads the library,
        and sets up the shared grid on the GPU.

        Args:
            grid (np.ndarray): The grid on which the wavelets are defined.
        """
        source_dir = os.path.dirname(os.path.abspath(__file__))
        lib_name = "wavelet_fast.so" if platform.system() == "Linux" else "wavelet_fast.dll"
        lib_path = os.path.join(source_dir, lib_name)
        
        if not os.path.exists(lib_path):
            source_path = os.path.join(source_dir, "wavelet_fast.cu")
            _compile_cuda_source(lib_path, source_path)

        self.lib = ctypes.CDLL(lib_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This class requires a GPU.")

        self._define_c_functions()

        self.grid = grid.astype(np.float32)
        self.grid_len = len(grid)
        self.grid_min = grid[0]
        self.grid_max = grid[-1]

        # Call the one-time grid setup function
        c_float_p = ctypes.POINTER(ctypes.c_float)
        h_grid_ptr = self.grid.ctypes.data_as(c_float_p)
        self.lib.setup_grid(h_grid_ptr, self.grid_len)
        print("CUDA device grid setup complete.")

    def _define_c_functions(self):
        """Defines the argument and return types for the C functions."""
        c_float_p = ctypes.POINTER(ctypes.c_float)
        
        self.lib.setup_grid.argtypes = [c_float_p, ctypes.c_int]
        self.lib.setup_grid.restype = None

        self.lib.evaluate_wavelet_fast_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, # d_phi, d_psi
            ctypes.c_float, ctypes.c_float, ctypes.c_int,
            ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_uint,
            ctypes.c_void_p
        ]
        self.lib.evaluate_wavelet_fast_async.restype = None

    def generate_batch(self, batch_params: list[dict]):
        """
        Generates a batch of datasets in parallel using CUDA streams.

        Args:
            batch_params (list[dict]): Each dict defines one task and must contain:
                                      'N' (int): Number of points.
                                      'd' (int): Dimension.
                                      'alpha' (float): Besov regularity.
                                      'J' (int): Max wavelet scale.
                                      'C0' (float): Scaling constant.
        
        Returns:
            list[torch.Tensor]: A list of generated datasets as GPU tensors.
        """
        num_streams = len(batch_params)
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        d_X_batch = [torch.randn(p['N'], p['d'], device=self.device) for p in batch_params]
        d_y_batch = [torch.empty(p['N'], device=self.device) for p in batch_params]
        
        # Keep references to wavelet tensors to prevent premature garbage collection
        wavelet_tensors = []

        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                params = batch_params[i]
                
                # Generate, resample, and transfer wavelet functions for this stream
                wavelet = pywt.Wavelet('db6')
                phi_pywt, psi_pywt, x_pywt = wavelet.wavefun(level=10)
                
                phi_resampled = np.interp(self.grid, x_pywt, phi_pywt).astype(np.float32)
                psi_resampled = np.interp(self.grid, x_pywt, psi_pywt).astype(np.float32)

                d_phi = torch.from_numpy(phi_resampled).to(self.device)
                d_psi = torch.from_numpy(psi_resampled).to(self.device)
                wavelet_tensors.append((d_phi, d_psi)) # Keep alive

                # Get pointers for the C function
                d_X_ptr = d_X_batch[i].data_ptr()
                d_y_ptr = d_y_batch[i].data_ptr()
                d_phi_ptr = d_phi.data_ptr()
                d_psi_ptr = d_psi.data_ptr()
                
                seed = random.randint(0, 2**32 - 1)
                
                # Launch the CUDA kernel asynchronously
                self.lib.evaluate_wavelet_fast_async(
                    d_X_ptr, d_y_ptr, params['N'], params['d'],
                    d_phi_ptr, d_psi_ptr,
                    self.grid_min, self.grid_max, self.grid_len,
                    params['alpha'], params['J'], params['C0'], seed,
                    ctypes.c_void_p(streams[i].cuda_stream)
                )

        torch.cuda.synchronize()
        
        print(f"Successfully generated {num_streams} datasets in parallel.")
        return d_y_batch

if __name__ == '__main__':
    # --- 1. Define the shared grid for all wavelet functions ---
    GRID_LEN = 4096
    # A wider grid is better to support different wavelet shapes
    grid = np.linspace(-8.0, 8.0, GRID_LEN, dtype=np.float32)
    
    # --- 2. Instantiate the Generator ---
    try:
        generator = WaveletGenerator(grid)

        # --- 3. Define Parameters for a Batch of 16 Datasets ---
        N_DATASETS = 16
        batch_generation_params = [
            {
                'N': 2048,
                'd': 2,
                'alpha': 1.5,
                'J': 6,
                'C0': 1.0,
            }
            for _ in range(N_DATASETS)
        ]

        # --- 4. Generate the Batch ---
        generated_datasets = generator.generate_batch(batch_generation_params)

        for i, dataset in enumerate(generated_datasets):
            params = batch_generation_params[i]
            print(f"Dataset {i} ({params['wavelet']}): Tensor of shape {dataset.shape} on device {dataset.device}")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")



