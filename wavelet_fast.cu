// Ultra-fast CUDA implementation with PRECOMPUTED coefficients and STREAMING
// Compile to shared library: nvcc -O3 -arch=sm_70 -shared -o wavelet_fast.dll wavelet_fast.cu -Xcompiler -fPIC

#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define THREADS_PER_BLOCK 256
#define MAX_DIM 4
#define MAX_GRID_SIZE 4096

// The grid is assumed to be constant for all calls
__constant__ float d_grid[MAX_GRID_SIZE];

// Fast linear interpolation
__device__ __forceinline__ float fast_interp(float t, const float* func, float grid_min, float grid_max, int grid_len) {
    if (t < grid_min || t > grid_max) return 0.0f;
    
    int left = 0;
    int right = grid_len - 1;
    
    #pragma unroll 8
    while (right - left > 1) {
        int mid = (left + right) >> 1;
        if (d_grid[mid] <= t) left = mid;
        else right = mid;
    }
    
    float w = (t - d_grid[left]) / (d_grid[right] - d_grid[left]);
    return func[left] * (1.0f - w) + func[right] * w;
}

// Compute flat index for coefficient lookup
__device__ __forceinline__ int compute_coeff_index(int j, const int* k_vec, 
                                                   int wavelet_mask, 
                                                   const int* offsets,
                                                   const int* k_ranges,
                                                   int d, int mask_count) {
    // Each (j, mask) combination has its own block of coefficients
    int block_offset = offsets[j * mask_count + wavelet_mask];
    
    // Compute position within block based on k_vec
    int idx = 0;
    int stride = 1;
    for (int i = 0; i < d; i++) {
        int k_relative = k_vec[i] - k_ranges[j * d * 2 + i * 2];  // k - k_min
        idx += k_relative * stride;
        stride *= (k_ranges[j * d * 2 + i * 2 + 1] - k_ranges[j * d * 2 + i * 2] + 1);
    }
    
    return block_offset + idx;
}

__device__ __forceinline__ bool advance_k_vector(int* k_vec,
                                                 const int* k_min,
                                                 const int* k_max,
                                                 int dims) {
    for (int i = 0; i < dims; i++) {
        k_vec[i]++;
        if (k_vec[i] <= k_max[i]) {
            return true;
        }
        k_vec[i] = k_min[i];
    }
    return false;
}

// Main kernel - coefficients are precomputed and stored in global memory
__global__ void evaluate_wavelet_precomputed(const float* __restrict__ X,
                                             const float* __restrict__ coeffs,
                                             const int* __restrict__ coeff_offsets,
                                             const int* __restrict__ k_ranges,
                                             const float* __restrict__ d_phi,
                                             const float* __restrict__ d_psi,
                                             float* __restrict__ y, int N,
                                             int d, int J, int mask_count,
                                             float grid_min, float grid_max, int grid_len) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    float result = 0.0f;
    float x_vec[MAX_DIM];
    
    #pragma unroll
    for (int i = 0; i < MAX_DIM; i++) {
        if (i < d) x_vec[i] = X[n * d + i];
    }
    
    int k_vec[MAX_DIM];
    int k_min[MAX_DIM], k_max[MAX_DIM];

    if (d == 0) {
        y[n] = 0.0f;
        return;
    }

    // Mother wavelets (j=0, mask=0)
    #pragma unroll
    for (int i = 0; i < MAX_DIM; i++) {
        if (i < d) {
            k_min[i] = k_ranges[i * 2];
            k_max[i] = k_ranges[i * 2 + 1];
            k_vec[i] = k_min[i];
        }
    }

    while (true) {
        float prod = 1.0f;
        bool skip = false;
        for (int i = 0; i < d; i++) {
            float t = x_vec[i] - (float)k_vec[i];
            float val = fast_interp(t, d_phi, grid_min, grid_max, grid_len);
            if (val == 0.0f) {
                skip = true;
                break;
            }
            prod *= val;
        }

        if (!skip) {
            int idx = compute_coeff_index(0, k_vec, 0, coeff_offsets, k_ranges, d, mask_count);
            result += coeffs[idx] * prod;
        }

        if (!advance_k_vector(k_vec, k_min, k_max, d)) {
            break;
        }
    }

    // Detail wavelets
    for (int j = 1; j <= J; j++) {
        float two_j = powf(2.0f, (float)j);
        float two_j_half = powf(2.0f, 0.5f * j);

        for (int i = 0; i < d; i++) {
            k_min[i] = k_ranges[j * d * 2 + i * 2];
            k_max[i] = k_ranges[j * d * 2 + i * 2 + 1];
        }

        for (int mask = 1; mask < mask_count; mask++) {
            for (int i = 0; i < d; i++) {
                k_vec[i] = k_min[i];
            }

            while (true) {
                float prod = 1.0f;
                bool skip = false;

                for (int i = 0; i < d; i++) {
                    float t = two_j * x_vec[i] - (float)k_vec[i];
                    const float* func = (mask & (1 << i)) ? d_psi : d_phi;
                    float val = fast_interp(t, func, grid_min, grid_max, grid_len) * two_j_half;
                    if (val == 0.0f) {
                        skip = true;
                        break;
                    }
                    prod *= val;
                }

                if (!skip) {
                    int idx = compute_coeff_index(j, k_vec, mask, coeff_offsets, k_ranges, d, mask_count);
                    result += coeffs[idx] * prod;
                }

                if (!advance_k_vector(k_vec, k_min, k_max, d)) {
                    break;
                }
            }
        }
    }
    y[n] = result;
}

__global__ void scale_coefficients_kernel(float* coeffs, const int* offsets,
                                         float alpha, float C0, int J, int d,
                                         int max_masks, int total_coeffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_coeffs) return;

    int j = -1;
    for (int current_j = 0; current_j <= J; ++current_j) {
        int start_mask = (current_j == 0) ? 0 : 1;
        int end_mask = (current_j == 0) ? 1 : max_masks;
        for (int current_mask = start_mask; current_mask < end_mask; ++current_mask) {
            int current_offset = offsets[current_j * max_masks + current_mask];
            if (current_offset == -1) continue;

            int next_offset;
            bool found_next = false;
            for (int next_j = current_j; next_j <= J; ++next_j) {
                int next_start_mask = (next_j == current_j) ? current_mask + 1 : 1;
                for (int next_mask = next_start_mask; next_mask < max_masks; ++next_mask) {
                    int temp_offset = offsets[next_j * max_masks + next_mask];
                    if (temp_offset != -1) {
                        next_offset = temp_offset;
                        found_next = true;
                        break;
                    }
                }
                if (found_next) break;
            }
            if (!found_next) {
                next_offset = total_coeffs;
            }
            
            if (idx >= current_offset && idx < next_offset) {
                j = current_j;
                break;
            }
        }
        if (j != -1) break;
    }

    if (j != -1) {
        float scale = (j == 0) ? (C0 * powf(2.0f, -0.5f * d)) :
                                 (C0 * powf(2.0f, -(alpha + d / 2.0f) * j));
        coeffs[idx] = scale * (coeffs[idx] * 2.0f - 1.0f);
    }
}

// Host-side coefficient generation
extern "C" {

typedef struct {
    float* d_coeffs;
    int* d_coeff_offsets;
    int* d_k_ranges;
    int total_coeffs;
} WaveletCoeffs;

// Setup function to copy grid data to constant memory
void setup_grid(const float* h_grid, int grid_len) {
    cudaMemcpyToSymbol(d_grid, h_grid, grid_len * sizeof(float));
}

WaveletCoeffs* precompute_coefficients_async(int d, float alpha, int J, float C0,
                                             uint32_t seed, int K_side,
                                             cudaStream_t stream) {
    WaveletCoeffs* wc = (WaveletCoeffs*)malloc(sizeof(WaveletCoeffs));
    
    int max_masks = (1 << d);
    int total_configs = (J + 1) * max_masks;
    
    int* h_offsets = (int*)malloc(total_configs * sizeof(int));
    int* h_k_ranges = (int*)malloc((J + 1) * d * 2 * sizeof(int));
    for (int i = 0; i < total_configs; i++) { h_offsets[i] = -1; }
    
    int total = 0;
    for (int j = 0; j <= J; j++) {
        int translations = 1 << j;
        for (int i = 0; i < d; i++) {
            int k_min = -K_side;
            int k_max = (translations - 1) + K_side;
            h_k_ranges[j * d * 2 + i * 2] = k_min;
            h_k_ranges[j * d * 2 + i * 2 + 1] = k_max;
        }
        int count_per_mask = 1;
        for (int i = 0; i < d; i++) {
            count_per_mask *= (h_k_ranges[j * d * 2 + i * 2 + 1] - h_k_ranges[j * d * 2 + i * 2] + 1);
        }
        int start_mask = (j == 0) ? 0 : 1;
        int end_mask = (j == 0) ? 1 : max_masks;
        for (int mask = start_mask; mask < end_mask; mask++) {
            h_offsets[j * max_masks + mask] = total;
            total += count_per_mask;
        }
    }
    wc->total_coeffs = total;
    
    cudaMallocAsync(&wc->d_coeffs, total * sizeof(float), stream);
    cudaMallocAsync(&wc->d_coeff_offsets, total_configs * sizeof(int), stream);
    cudaMallocAsync(&wc->d_k_ranges, (J + 1) * d * 2 * sizeof(int), stream);
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(gen, stream);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, wc->d_coeffs, total);
    
    cudaMemcpyAsync(wc->d_coeff_offsets, h_offsets, total_configs * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(wc->d_k_ranges, h_k_ranges, (J + 1) * d * 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
    
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scale_coefficients_kernel<<<blocks, threads, 0, stream>>>(
        wc->d_coeffs, wc->d_coeff_offsets, alpha, C0, J, d, max_masks, total
    );
    
    curandDestroyGenerator(gen);
    free(h_offsets);
    free(h_k_ranges);
    
    return wc;
}

// Main evaluation function.
void evaluate_wavelet_fast_async(float* d_X, float* d_y, int N, int d,
                                 const float* d_phi, const float* d_psi,
                                 float grid_min, float grid_max, int grid_len,
                                 float alpha, int J, float C0, uint32_t seed,
                                 cudaStream_t stream) {
    
    int K_side = (int)ceilf(grid_max - grid_min);
    int mask_count = 1 << d;

    WaveletCoeffs* wc = precompute_coefficients_async(d, alpha, J, C0, seed, K_side, stream);
    
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    evaluate_wavelet_precomputed<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_X, wc->d_coeffs, wc->d_coeff_offsets, wc->d_k_ranges,
        d_phi, d_psi, d_y, N, d, J, mask_count, grid_min, grid_max, grid_len
    );
    
    cudaFreeAsync(wc->d_coeffs, stream);
    cudaFreeAsync(wc->d_coeff_offsets, stream);
    cudaFreeAsync(wc->d_k_ranges, stream);
    free(wc);
}

} // extern "C"
