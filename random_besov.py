import ctypes
import numpy as np
import pywt
import os
import shutil
import subprocess
import platform
import sys
import torch

MAX_GRID_SIZE = 4096


def _library_basename():
    """Returns the platform-specific shared library name."""
    system = platform.system()
    if system == "Windows":
        return "wavelet_fast.dll"
    elif system == "Darwin":
        return "libwavelet_fast.dylib"
    else:
        return "wavelet_fast.so"

def _find_cuda_home():
    """Attempts to locate the CUDA installation path in a cross-platform way."""
    # 1. Most reliable: Find `nvcc` in the PATH
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # The CUDA root is typically two directories above the 'bin' where nvcc resides
        return os.path.dirname(os.path.dirname(nvcc_path))

    # 2. Check common environment variables
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_ROOT")
    if cuda_home and os.path.exists(cuda_home):
        return cuda_home

    # 3. Platform-specific fallbacks
    system = platform.system()
    if system == "Windows":
        # Check CUDA_PATH, common on Windows
        cuda_path_env = os.environ.get("CUDA_PATH")
        if cuda_path_env and os.path.exists(cuda_path_env):
            return cuda_path_env
        # Check default installation path
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        default_path = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.exists(default_path):
            # Find the latest version installed
            versions = [d for d in os.listdir(default_path) if os.path.isdir(os.path.join(default_path, d))]
            if versions:
                return os.path.join(default_path, sorted(versions)[-1])
    elif system == "Linux":
        # Check default installation path
        if os.path.exists("/usr/local/cuda"):
            return "/usr/local/cuda"
            
    return None # Return None if no path was found

def _get_library_path():
    """
    Returns the path to the pre-compiled CUDA library.
    Assumes the library is in the same directory as this script.
    """
    lib_name = _library_basename()
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(lib_dir, lib_name)

    if not os.path.exists(lib_path):
        if platform.system() == "Linux":
            src_path = os.path.join(lib_dir, "wavelet_fast.cu")
            if not os.path.exists(src_path):
                raise FileNotFoundError(
                    f"CUDA source not found at: {src_path}\n"
                    "Cannot compile the shared library."
                )
            nvcc_path = shutil.which("nvcc")
            if not nvcc_path:
                raise FileNotFoundError(
                    "nvcc not found in PATH; cannot compile wavelet_fast.cu automatically."
                )

            cmd = [
                nvcc_path,
                "-arch=sm_80",
                "-O3",
                "-o",
                lib_path,
                "-shared",
                "-Xcompiler",
                "-fPIC,-O3",
                src_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(lib_path):
                raise RuntimeError(
                    "Failed to compile wavelet_fast.cu with nvcc.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
        else:
            raise FileNotFoundError(
                f"Shared library not found at: {lib_path}\n"
                f"If you are compiling manually, make sure the compiled library '{lib_name}' "
                "is placed in the same directory as this Python script."
            )
    return lib_path




class FastWaveletBatchGenerator:
    """
    Ultra-fast CUDA wavelet evaluator for generating batches of random functions.
    """
    
    def __init__(self, wavelet='db6', cascade_level=8):
        # 1. Get wavelet data from PyWavelets
        w = pywt.Wavelet(wavelet)
        phi, psi, grid = w.wavefun(level=cascade_level)
        
        self.phi = np.ascontiguousarray(phi.astype(np.float32))
        self.psi = np.ascontiguousarray(psi.astype(np.float32))
        self.grid = np.ascontiguousarray(grid.astype(np.float32))

        if self.grid.size > MAX_GRID_SIZE:
            raise ValueError(f"Wavelet grid has {self.grid.size} points, but the CUDA kernel "
                             f"was built with MAX_GRID_SIZE={MAX_GRID_SIZE}.")
        
        # 2. Get the path to the pre-compiled CUDA library
        lib_path = _get_library_path()
        
        # 3. Pre-load CUDA dependencies to ensure symbols are available for our library.
        #    This is platform-specific.
        if platform.system() == "Linux":
            # On Linux, we need to manually load shared libraries that our .so file depends on,
            # especially if the CUDA lib path is not in the system's default search paths or LD_LIBRARY_PATH.
            loaded_deps = False
            try:
                # First, try loading libraries by name, assuming LD_LIBRARY_PATH is set.
                ctypes.CDLL("libcurand.so", mode=ctypes.RTLD_GLOBAL)
                ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)
                loaded_deps = True
            except OSError:
                # If that fails, try to find the CUDA installation and load from its lib directory.
                cuda_home = _find_cuda_home()
                if cuda_home:
                    lib_dir = os.path.join(cuda_home, 'lib64')
                    if not os.path.isdir(lib_dir):
                        lib_dir = os.path.join(cuda_home, 'lib')  # Fallback
                    
                    try:
                        ctypes.CDLL(os.path.join(lib_dir, "libcurand.so"), mode=ctypes.RTLD_GLOBAL)
                        ctypes.CDLL(os.path.join(lib_dir, "libcudart.so"), mode=ctypes.RTLD_GLOBAL)
                        loaded_deps = True
                    except OSError:
                        pass  # Failure will be handled by the final check.

            if not loaded_deps:
                print(
                    "Warning: Could not pre-load CUDA libraries 'libcurand.so' or 'libcudart.so'.\n"
                    "The script will still attempt to load 'wavelet_fast.so', but it may fail with an 'undefined symbol' error.\n"
                    "To fix this, please ensure your CUDA library path (e.g., /usr/local/cuda/lib64) is in the LD_LIBRARY_PATH environment variable."
                )
        
        elif platform.system() == "Windows" and hasattr(os, 'add_dll_directory'):
            # On Windows, we add the CUDA bin directory to the DLL search path.
            # This is critical for finding dependencies like curand64_xx.dll.
            cuda_home = _find_cuda_home()
            if cuda_home:
                cuda_bin_path = os.path.join(cuda_home, "bin")
                if os.path.isdir(cuda_bin_path):
                    os.add_dll_directory(cuda_bin_path)
            else:
                print("Warning: Could not find CUDA installation path. Loading the library might fail "
                      "if its dependencies (e.g., curand.dll) are not in the system PATH.")

        # 4. Load the compiled library
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            if platform.system() == "Windows":
                 raise OSError(
                    f"Could not load the compiled library at {lib_path}. This often happens if a CUDA dependency "
                    f"(like 'curand64_xx.dll') is not found. Ensure the CUDA toolkit is installed correctly "
                    f"and its 'bin' directory is in the system's PATH environment variable."
                ) from e
            raise e
        
        # 5. Define the batch generation function signature for ctypes
        self.lib.generate_wavelet_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # h_X_concat
            ctypes.POINTER(ctypes.c_float),  # h_Y_concat
            ctypes.POINTER(ctypes.c_int),    # h_N_values
            ctypes.POINTER(ctypes.c_int),    # h_offsets
            ctypes.c_int,                    # total_points
            ctypes.c_int,                    # d
            ctypes.c_float,                  # alpha
            ctypes.c_int,                    # J
            ctypes.c_float,                  # C0
            ctypes.c_int,                    # batch_size
            ctypes.POINTER(ctypes.c_float),  # h_phi
            ctypes.POINTER(ctypes.c_float),  # h_psi
            ctypes.POINTER(ctypes.c_float),  # h_grid
            ctypes.c_int,                    # grid_len
            ctypes.c_int,                    # max_N
            ctypes.c_ulonglong,              # seed (0 uses time)
        ]
        self.lib.generate_wavelet_batch.restype = ctypes.c_int

        self.lib.cleanup_wavelet_resources.argtypes = []
        self.lib.cleanup_wavelet_resources.restype = ctypes.c_int
    
    def cleanup(self):
        """
        Explicitly releases persistent CUDA resources held by the library.
        It's good practice to call this when you're done with the generator.
        """
        if self.lib:
            self.lib.cleanup_wavelet_resources()
            self.lib = None

    def __del__(self):
        """Ensures cleanup is called when the object is garbage collected."""
        self.cleanup()

    def generate_batch(self, X_list, params):
        """
        Generates a batch of random functions, each on its own input grid.
        
        Args:
            X_list (list of np.ndarray): A list of input points arrays. Each array
                can have a different number of points (N), but must have the
                same dimension (d) as specified in params. E.g., [X1, X2] 
                where X1.shape=(N1, d) and X2.shape=(N2, d).
            params (dict): A dictionary containing the shared parameters for the
                entire batch: {'d', 'alpha', 'J', 'C0', 'seed'}. If 'seed' is 0
                or missing, the CUDA code uses time().
        """
        if not self.lib:
            raise RuntimeError("The CUDA library has been unloaded. Please create a new generator instance.")
        if not X_list:
            return []
            
        # Extract shared parameters from the params dictionary.
        d = params['d']
        J = params['J']
        alpha = params.get('alpha', 1.0)
        C0 = params.get('C0', 1.0)
        seed = int(params.get('seed', 0) or 0)

        if d < 1 or d > 4:
            raise ValueError(f"d must be between 1 and 4 (MAX_DIM in CUDA), got {d}")
        
        # Prepare data for concatenated input
        processed_X_list = []
        for i, X in enumerate(X_list):
            X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
            if X_arr.ndim == 1: X_arr = X_arr.reshape(-1, d)
            if X_arr.shape[1] != d:
                raise ValueError(f"X at index {i} must have shape (N, {d}), got {X_arr.shape}")
            processed_X_list.append(X_arr)

        batch_size = len(X_list)
        N_values = np.array([x.shape[0] for x in processed_X_list], dtype=np.int32)
        total_points = np.sum(N_values)
        max_N = np.max(N_values) if N_values.size > 0 else 0
        
        # Offsets are the cumulative sum of points, starting with 0
        offsets = np.concatenate(([0], np.cumsum(N_values[:-1]))).astype(np.int32)

        X_concat = np.vstack(processed_X_list) if processed_X_list else np.array([], dtype=np.float32).reshape(0, d)
                    
        # Prepare output array
        Y_concat = np.ascontiguousarray(np.zeros(total_points, dtype=np.float32))
        
        # Call the CUDA function
        ret_code = self.lib.generate_wavelet_batch(
            X_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            Y_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N_values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            total_points,
            d,
            alpha,
            J,
            C0,
            batch_size,
            self.phi.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.psi.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(self.grid),
            max_N,
            seed
        )

        # Check the return code from the C++ function
        if ret_code != 0:
            raise RuntimeError(
                f"The CUDA function 'generate_wavelet_batch' failed with error code {ret_code}. "
                "This is likely due to a CUDA error, such as running out of GPU memory. "
                "Check the console output of your Jupyter notebook server for a more detailed 'CUDA Error' message."
            )
        
        # Split the concatenated output back into a list of arrays
        if total_points > 0:
            return np.split(Y_concat, offsets[1:])
        else:
            return []

