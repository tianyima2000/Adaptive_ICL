#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Helper macro for checking CUDA calls (expects `status` variable and `cleanup` label in scope)
#define CUDA_CHECK(call)                                                                 \
do {                                                                                     \
    cudaError_t err = call;                                                              \
    if (err != cudaSuccess) {                                                            \
        fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n", __FILE__, __LINE__,             \
                cudaGetErrorName(err), cudaGetErrorString(err));                         \
        fflush(stderr);                                                                  \
        status = static_cast<int>(err);                                                  \
        goto cleanup;                                                                    \
    }                                                                                    \
} while (0)

// Helper macro for checking cuRAND calls (expects `status` variable and `cleanup` label in scope)
#define CURAND_CHECK(call)                                                               \
do {                                                                                     \
    curandStatus_t status_curand = call;                                                 \
    if (status_curand != CURAND_STATUS_SUCCESS) {                                        \
        fprintf(stderr, "cuRAND Error at %s:%d - status %d\n", __FILE__, __LINE__,        \
                status_curand);                                                          \
        fflush(stderr);                                                                  \
        status = -1000 - static_cast<int>(status_curand);                                \
        goto cleanup;                                                                    \
    }                                                                                    \
} while (0)


// --- WINDOWS DLL EXPORT FIX ---
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif
// ------------------------------

#define THREADS_PER_BLOCK 256
#define MAX_DIM 4
#define MAX_GRID_SIZE 4096

// Constant memory for wavelet data (shared across all functions in a batch)
__constant__ float d_phi[MAX_GRID_SIZE];
__constant__ float d_psi[MAX_GRID_SIZE];
__constant__ float d_grid[MAX_GRID_SIZE];
__constant__ float d_grid_min;
__constant__ float d_grid_max;
__constant__ float d_support_min;
__constant__ float d_support_max;
__constant__ int d_grid_len;
__constant__ int d_K_side;

// Shared wavelet parameters (constant across the batch)
__constant__ float d_alpha;
__constant__ int d_J;
__constant__ float d_C0;
__constant__ int d_d;

// Per-function seeds are now generated on-the-fly inside the host function.

// --- Kernel to compute metadata ONCE for the whole batch ---
__global__ void compute_shared_metadata_kernel(
    int max_total_configs,
    int* __restrict__ all_coeff_offsets,
    int* __restrict__ all_k_ranges,
    int* __restrict__ total_coeffs_per_func)
{
    // This kernel is launched with a single block. It calculates the metadata
    // for func_idx = 0. This metadata is then broadcast to all other functions.
    if (threadIdx.x == 0) {
        int d = d_d;
        int J = d_J;
        int K_side = d_K_side;
        int max_masks = 1 << d;

        int* h_offsets = &all_coeff_offsets[0]; // Writing to the first slot
        int* h_k_ranges = &all_k_ranges[0];     // Writing to the first slot

        int total = 0;
        for (int j = 0; j <= J; j++) {
            int translations = 1 << j;
            int count_per_mask = 1;
            for (int i = 0; i < d; i++) {
                int k_min = -K_side;
                int k_max = (translations - 1) + K_side;
                h_k_ranges[j * d * 2 + i * 2] = k_min;
                h_k_ranges[j * d * 2 + i * 2 + 1] = k_max;
                count_per_mask *= (k_max - k_min + 1);
            }
            
            int start_mask = (j == 0) ? 0 : 1;
            int end_mask = (j == 0) ? 1 : max_masks;
            for (int mask = start_mask; mask < end_mask; mask++) {
                h_offsets[j * max_masks + mask] = total;
                total += count_per_mask;
            }
        }
        total_coeffs_per_func[0] = total; // Writing to the first slot
    }
}

// --- Kernel to broadcast the shared metadata from function 0 to all others ---
__global__ void broadcast_metadata_kernel(
    int batch_size,
    int max_total_configs,
    int k_ranges_size,
    int* __restrict__ all_coeff_offsets,
    int* __restrict__ all_k_ranges,
    int* __restrict__ total_coeffs_per_func)
{
    int func_idx = blockIdx.x + 1; // Start from function 1
    if (func_idx >= batch_size) return;

    // Source pointers (from function 0)
    const int* src_offsets = &all_coeff_offsets[0];
    const int* src_k_ranges = &all_k_ranges[0];
    const int src_total_coeffs = total_coeffs_per_func[0];

    // Destination pointers
    int* dst_offsets = &all_coeff_offsets[func_idx * max_total_configs];
    int* dst_k_ranges = &all_k_ranges[func_idx * k_ranges_size];

    // Copy data in parallel
    for (int i = threadIdx.x; i < max_total_configs; i += blockDim.x) {
        dst_offsets[i] = src_offsets[i];
    }
    
    for (int i = threadIdx.x; i < k_ranges_size; i += blockDim.x) {
        dst_k_ranges[i] = src_k_ranges[i];
    }

    if (threadIdx.x == 0) {
        total_coeffs_per_func[func_idx] = src_total_coeffs;
    }
}

// --- Kernel to scale the random coefficients ---
__global__ void scale_coefficients_kernel(
    int batch_size,
    int max_total_coeffs,
    int max_total_configs,
    float* __restrict__ all_coeffs,
    const int* __restrict__ all_coeff_offsets,
    const int* __restrict__ all_k_ranges,
    const int* __restrict__ total_coeffs_per_func)
{
    int func_idx = blockIdx.x;
    if (func_idx >= batch_size) return;

    int d = d_d;
    int J = d_J;
    int max_masks = 1 << d;

    const int* h_offsets = &all_coeff_offsets[func_idx * max_total_configs];
    const int* h_k_ranges = &all_k_ranges[func_idx * (J + 1) * d * 2];
    
    int total_coeffs = total_coeffs_per_func[func_idx];
    float* h_coeffs = &all_coeffs[func_idx * max_total_coeffs];

    for (int j = 0; j <= J; j++) {

        float scale;
        if (j == 0) {
            scale = d_C0 * powf(2.0f, -0.5f * d);
        } else {
            scale = d_C0 * powf(2.0f, -(d_alpha + d / 2.0f) * j);
        }

        int count_per_mask = 1;
        for (int i = 0; i < d; ++i) {
            count_per_mask *= (h_k_ranges[j * d * 2 + i * 2 + 1]
                             - h_k_ranges[j * d * 2 + i * 2] + 1);
        }

        int start_mask = (j == 0) ? 0 : 1;
        int end_mask   = (j == 0) ? 1 : max_masks;

        for (int mask = start_mask; mask < end_mask; mask++) {
            int offset = h_offsets[j * max_masks + mask];

            int idx = threadIdx.x;
            while (idx < count_per_mask) {
                int an_idx = offset + idx;
                if (an_idx < total_coeffs) {
                    float u = h_coeffs[an_idx];  // u ~ U[0,1] from cuRAND

                    if (j == 0) {
                        // Level 0: C0 * 2^{-d/2} * U[0,1]
                        h_coeffs[an_idx] = scale * u;
                    } else {
                        // Levels j >= 1: C0 * 2^{-(alpha + d/2) j} * U[-1,1]
                        h_coeffs[an_idx] = scale * (2.0f * u - 1.0f);
                    }
                }
                idx += blockDim.x;
            }
        }
    }
}



// --- Batched Wavelet Evaluation Kernel ---

__device__ __forceinline__ float fast_interp(float t, const float* func) {
    if (t < d_grid_min || t > d_grid_max) return 0.0f;
    int left = 0, right = d_grid_len - 1;
    #pragma unroll 8
    while (right - left > 1) {
        int mid = (left + right) >> 1;
        if (d_grid[mid] <= t) left = mid; else right = mid;
    }
    float w = (t - d_grid[left]) / (d_grid[right] - d_grid[left]);
    return func[left] * (1.0f - w) + func[right] * w;
}

__device__ __forceinline__ int compute_coeff_index(int j, const int* k_vec, int wavelet_mask, 
                                                   const int* offsets, const int* k_ranges,
                                                   int d, int max_masks) {
    int block_offset = offsets[j * max_masks + wavelet_mask];
    int idx = 0, stride = 1;
    for (int i = 0; i < d; i++) {
        int k_relative = k_vec[i] - k_ranges[j * d * 2 + i * 2];
        idx += k_relative * stride;
        stride *= (k_ranges[j * d * 2 + i * 2 + 1] - k_ranges[j * d * 2 + i * 2] + 1);
    }
    return block_offset + idx;
}

__device__ __forceinline__ bool advance_k_vector(int* k_vec, const int* k_min, const int* k_max, int dims) {
    for (int i = 0; i < dims; i++) {
        k_vec[i]++;
        if (k_vec[i] <= k_max[i]) return true;
        k_vec[i] = k_min[i];
    }
    return false;
}

__global__ void evaluate_wavelet_batch_kernel(
    const float* __restrict__ X_concat,
    const float* __restrict__ all_coeffs,
    const int* __restrict__ all_coeff_offsets,
    const int* __restrict__ all_k_ranges,
    int max_total_coeffs,
    int max_total_configs,
    float* __restrict__ Y_concat,
    const int* __restrict__ N_values,
    const int* __restrict__ offsets)
{
    int func_idx = blockIdx.x;
    int n = threadIdx.x + blockIdx.y * blockDim.x;

    int N_current_func = N_values[func_idx];
    if (n >= N_current_func) return;

    // Shared parameters are now read from constant memory
    int d = d_d;
    int J = d_J;
    int max_masks = 1 << d;

    int x_offset = offsets[func_idx];
    int y_offset = offsets[func_idx];

    const float* coeffs = &all_coeffs[func_idx * max_total_coeffs];
    const int* coeff_offsets = &all_coeff_offsets[func_idx * max_total_configs];
    const int* k_ranges = &all_k_ranges[func_idx * (J + 1) * d * 2];
    const float* x_vec_base = &X_concat[(x_offset + n) * d];
    float* y_out = &Y_concat[y_offset + n];

    float result = 0.0f;
    float x_vec[MAX_DIM];
    #pragma unroll
    for (int i = 0; i < d; i++) x_vec[i] = x_vec_base[i];

    int k_vec[MAX_DIM], k_min[MAX_DIM], k_max[MAX_DIM];

    // Mother wavelets (j=0)
    bool has_support = true;
    for (int i = 0; i < d; i++) {
        int domain_min = k_ranges[i * 2], domain_max = k_ranges[i * 2 + 1];
        int start = (int)ceilf(x_vec[i] - d_support_max);
        int stop = (int)floorf(x_vec[i] - d_support_min);
        k_min[i] = max(start, domain_min);
        k_max[i] = min(stop, domain_max);
        if (k_max[i] < k_min[i]) { has_support = false; break; }
        k_vec[i] = k_min[i];
    }

    if (has_support) {
        do {
            float prod = 1.0f;
            for (int i = 0; i < d; i++) {
                prod *= fast_interp(x_vec[i] - (float)k_vec[i], d_phi);
            }
            if (prod != 0.0f) {
                int idx = compute_coeff_index(0, k_vec, 0, coeff_offsets, k_ranges, d, max_masks);
                result += coeffs[idx] * prod;
            }
        } while (advance_k_vector(k_vec, k_min, k_max, d));
    }

    // Detail wavelets
    float two_j = 1.0f; // Starts at 2^0
    float two_j_half = 1.0f; // Starts at 2^0
    const float sqrt2 = sqrtf(2.0f);

    for (int j = 1; j <= J; j++) {
        two_j *= 2.0f;
        two_j_half *= sqrt2;

        has_support = true;
        for (int i = 0; i < d; i++) {
            int domain_min = k_ranges[j * d * 2 + i * 2], domain_max = k_ranges[j * d * 2 + i * 2 + 1];
            int start = (int)ceilf(x_vec[i] * two_j - d_support_max);
            int stop = (int)floorf(x_vec[i] * two_j - d_support_min);
            k_min[i] = max(start, domain_min);
            k_max[i] = min(stop, domain_max);
            if (k_max[i] < k_min[i]) { has_support = false; break; }
        }
        if (!has_support) continue;

        for (int mask = 1; mask < max_masks; mask++) {
            for(int i=0; i<d; ++i) k_vec[i] = k_min[i];
            do {
                float prod = 1.0f;
                for (int i = 0; i < d; i++) {
                    const float* func = (mask & (1 << i)) ? d_psi : d_phi;
                    prod *= fast_interp(two_j * x_vec[i] - (float)k_vec[i], func) * two_j_half;
                }
                if (prod != 0.0f) {
                    int idx = compute_coeff_index(j, k_vec, mask, coeff_offsets, k_ranges, d, max_masks);
                    result += coeffs[idx] * prod;
                }
            } while (advance_k_vector(k_vec, k_min, k_max, d));
        }
    }
    *y_out = result;
}

// --- Persistent CUDA Streams (reused, but generators recreated each time) ---
static cudaStream_t* g_streams = NULL;
static int g_num_streams = 0;

int initialize_cuda_streams(int num_streams) {
    if (g_streams != NULL) return 0; // Already initialized
    
    int status = 0;
    g_num_streams = num_streams;
    g_streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    if (g_streams == NULL) {
        fprintf(stderr, "Failed to allocate host memory for streams.\n");
        return -1;
    }
    for (int i = 0; i < num_streams; i++) g_streams[i] = NULL;
    
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&g_streams[i]));
    }
    return 0;

cleanup:
    if (g_streams != NULL) {
        for (int j = 0; j < num_streams; j++) {
            if (g_streams[j] != NULL) {
                cudaStreamDestroy(g_streams[j]);
            }
        }
        free(g_streams);
        g_streams = NULL;
        g_num_streams = 0;
    }
    return status;
}

int cleanup_cuda_streams() {
    if (g_streams == NULL) return 0;
    
    int status = 0;
    for (int i = 0; i < g_num_streams; i++) {
        cudaError_t err = cudaStreamDestroy(g_streams[i]);
        if (err != cudaSuccess && status == 0) {
            fprintf(stderr, "CUDA Error during stream destroy at %s:%d - %s: %s\n",
                    __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err));
            fflush(stderr);
            status = static_cast<int>(err);
        }
    }
    free(g_streams);
    g_streams = NULL;
    g_num_streams = 0;
    return status;
}

// --- Host-side Entry Point ---
extern "C" {

DLLEXPORT int generate_wavelet_batch(
    const float* h_X_concat,
    float* h_Y_concat,
    const int* h_N_values,
    const int* h_offsets,
    int total_points,
    int d,
    float alpha,
    int J,
    float C0,
    int batch_size,
    const float* h_phi,
    const float* h_psi,
    const float* h_grid,
    int grid_len,
    int max_N)
{
    int status = 0;
    float *d_X_concat = NULL, *d_Y_concat = NULL, *d_all_coeffs = NULL;
    int *d_all_coeff_offsets = NULL, *d_all_k_ranges = NULL, *d_total_coeffs_per_func = NULL;
    int *d_N_values = NULL, *d_offsets = NULL;
    int h_total_coeffs;
    curandGenerator_t* gens = NULL;
    int gens_created = 0;
    unsigned long long base_seed = time(NULL);

    // Validate dimension to avoid overrunning fixed-size stack buffers in kernels
    if (d < 1 || d > MAX_DIM) {
        fprintf(stderr, "Invalid dimension d=%d (MAX_DIM=%d)\n", d, MAX_DIM);
        return -1;
    }
    if (grid_len < 1 || grid_len > MAX_GRID_SIZE) {
        fprintf(stderr, "Invalid grid_len=%d (MAX_GRID_SIZE=%d)\n", grid_len, MAX_GRID_SIZE);
        return -1;
    }

    // Initialize persistent CUDA streams (only once per process)
    if (g_streams == NULL || g_num_streams == 0) {
        int num_streams = (batch_size < 32) ? batch_size : 32;
        status = initialize_cuda_streams(num_streams);
        if (status != 0) {
            return status;
        }
    }

    // Predeclare variables so cleanup goto does not bypass initialization (silences NVCC warnings)
    float grid_min = 0.0f, grid_max = 0.0f;
    int K_side = 0;
    float support_min = 0.0f, support_max = 0.0f;
    int max_masks = 0;
    int max_total_configs = 0;
    int max_total_coeffs = 0;
    dim3 grid_dim(1, 1);
    int k_ranges_size = 0;

    // 1. Copy wavelet and shared parameters to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_phi, h_phi, grid_len * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_psi, h_psi, grid_len * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_grid, h_grid, grid_len * sizeof(float)));
    
    grid_min = h_grid[0];
    grid_max = h_grid[grid_len-1];
    CUDA_CHECK(cudaMemcpyToSymbol(d_grid_min, &grid_min, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_grid_max, &grid_max, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_grid_len, &grid_len, sizeof(int)));
    
    K_side = (int)ceilf(grid_max - grid_min);
    CUDA_CHECK(cudaMemcpyToSymbol(d_K_side, &K_side, sizeof(int)));
    support_min = -static_cast<float>(K_side);
    support_max = static_cast<float>(K_side);
    CUDA_CHECK(cudaMemcpyToSymbol(d_support_min, &support_min, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_support_max, &support_max, sizeof(float)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha, &alpha, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_J, &J, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_C0, &C0, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_d, &d, sizeof(int)));

    // 2. Determine max buffer sizes
    max_masks = 1 << d;
    max_total_configs = (J + 1) * max_masks;
    k_ranges_size = (J + 1) * d * 2;
    
    // Correctly calculate the maximum possible number of coefficients
    max_total_coeffs = 0;
    for (int j = 0; j <= J; j++) {
        int count_per_mask = 1;
        int current_translations = 1 << j;
        for (int i = 0; i < d; i++) {
            count_per_mask *= ((current_translations - 1) + 2 * K_side + 1);
        }
        int num_masks_at_level = (j == 0) ? 1 : max_masks - 1;
        max_total_coeffs += num_masks_at_level * count_per_mask;
    }

    // 3. Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_X_concat, total_points * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_concat, total_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_coeffs, batch_size * max_total_coeffs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_coeff_offsets, batch_size * max_total_configs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_k_ranges, batch_size * k_ranges_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total_coeffs_per_func, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_N_values, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, batch_size * sizeof(int)));

    // 4. Copy inputs from host to device
    CUDA_CHECK(cudaMemcpy(d_X_concat, h_X_concat, total_points * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N_values, h_N_values, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // 5. Compute metadata once and broadcast
    compute_shared_metadata_kernel<<<1, THREADS_PER_BLOCK>>>(
        max_total_configs, d_all_coeff_offsets, d_all_k_ranges, d_total_coeffs_per_func
    );
    CUDA_CHECK(cudaPeekAtLastError());

    if (batch_size > 1) {
        broadcast_metadata_kernel<<<batch_size - 1, THREADS_PER_BLOCK>>>(
            batch_size, max_total_configs, k_ranges_size,
            d_all_coeff_offsets, d_all_k_ranges, d_total_coeffs_per_func
        );
        CUDA_CHECK(cudaPeekAtLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 6. Copy the single total_coeffs value back to host
    CUDA_CHECK(cudaMemcpy(&h_total_coeffs, d_total_coeffs_per_func, sizeof(int), cudaMemcpyDeviceToHost));

    // 7. Generate random numbers
    gens = (curandGenerator_t*)malloc(batch_size * sizeof(curandGenerator_t));
    if (gens == NULL) {
        fprintf(stderr, "Failed to allocate host memory for cuRAND generators.\n");
        status = -1;
        goto cleanup;
    }
    for (int i = 0; i < batch_size; i++) gens[i] = NULL;

    for (int func_idx = 0; func_idx < batch_size; func_idx++) {
        if (h_total_coeffs > max_total_coeffs) {
            fprintf(stderr, "Computed coefficient count %d exceeds buffer capacity %d\n", h_total_coeffs, max_total_coeffs);
            status = -1;
            goto cleanup;
        }
        float* d_func_coeffs = d_all_coeffs + func_idx * max_total_coeffs;
        int aligned_total = (h_total_coeffs % 2 == 0) ? h_total_coeffs : h_total_coeffs + 1;
        if (aligned_total > max_total_coeffs) aligned_total = max_total_coeffs;
        
        CURAND_CHECK(curandCreateGenerator(&gens[func_idx], CURAND_RNG_PSEUDO_DEFAULT));
        gens_created++;
        int stream_idx = func_idx % g_num_streams;
        CURAND_CHECK(curandSetStream(gens[func_idx], g_streams[stream_idx]));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gens[func_idx], base_seed + func_idx));
        CURAND_CHECK(curandGenerateUniform(gens[func_idx], d_func_coeffs, aligned_total));
    }

    for (int i = 0; i < g_num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(g_streams[i]));
    }
    
    // 8. Scale the coefficients
    scale_coefficients_kernel<<<batch_size, THREADS_PER_BLOCK>>>(
        batch_size, max_total_coeffs, max_total_configs,
        d_all_coeffs, d_all_coeff_offsets, d_all_k_ranges, d_total_coeffs_per_func
    );
    CUDA_CHECK(cudaPeekAtLastError());

    // 9. Evaluate all functions in the batch
    {
        int blocks_y = (max_N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (blocks_y > 0) {
            grid_dim = dim3(batch_size, blocks_y);
            evaluate_wavelet_batch_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(
                d_X_concat, d_all_coeffs, d_all_coeff_offsets, d_all_k_ranges,
                max_total_coeffs, max_total_configs, d_Y_concat, d_N_values, d_offsets
            );
            CUDA_CHECK(cudaPeekAtLastError());
        }
    }

    // 10. Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_Y_concat, d_Y_concat, total_points * sizeof(float), cudaMemcpyDeviceToHost));

cleanup:
    if (gens != NULL) {
        for (int i = 0; i < gens_created; i++) {
            if (gens[i] != NULL) {
                curandDestroyGenerator(gens[i]);
            }
        }
        free(gens);
    }
    // No need to free h_total_coeffs_per_func, it's a single int on the stack
    // Free device memory best-effort; record first error if not already set
    auto free_device = [&](void* ptr) {
        if (ptr != NULL) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess && status == 0) {
                status = static_cast<int>(err);
            }
        }
    };
    free_device(d_X_concat);
    free_device(d_Y_concat);
    free_device(d_all_coeffs);
    free_device(d_all_coeff_offsets);
    free_device(d_all_k_ranges);
    free_device(d_total_coeffs_per_func);
    free_device(d_N_values);
    free_device(d_offsets);

    return status;
}

// Optional cleanup function
DLLEXPORT int cleanup_wavelet_resources() {
    return cleanup_cuda_streams();
}

} // extern "C"