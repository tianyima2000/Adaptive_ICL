#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

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

// Per-function parameters
typedef struct {
    float alpha;
    int J;
    float C0;
    uint32_t seed;
    int d; // Dimension
} WaveletParams;

// --- Fused Kernel for Metadata + Scaling ---
__global__ void compute_metadata_and_scale_kernel(
    WaveletParams* params,
    int batch_size,
    int max_total_coeffs,
    int max_total_configs,
    float* __restrict__ all_coeffs,
    int* __restrict__ all_coeff_offsets,
    int* __restrict__ all_k_ranges,
    int* __restrict__ total_coeffs_per_func,
    int max_J,
    bool compute_metadata,
    bool scale_coefficients)
{
    int func_idx = blockIdx.x;
    if (func_idx >= batch_size) return;

    WaveletParams p = params[func_idx];
    int d = p.d;
    int J = p.J;
    int K_side = d_K_side;
    int max_masks = 1 << d;

    int* h_offsets = &all_coeff_offsets[func_idx * max_total_configs];
    int* h_k_ranges = &all_k_ranges[func_idx * (max_J + 1) * d * 2];

    if (compute_metadata) {
        if (threadIdx.x == 0) {
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
            total_coeffs_per_func[func_idx] = total;
        }
    }
    
    // Synchronize to ensure metadata is visible to all threads before scaling
    __syncthreads();

    if (scale_coefficients) {
        int total_coeffs = total_coeffs_per_func[func_idx];
        float* h_coeffs = &all_coeffs[func_idx * max_total_coeffs];

        // Parallel scaling of coefficients
        for (int j = 0; j <= J; j++) {
            float scale = (j == 0) ? (p.C0 * powf(2.0f, -0.5f * d)) :
                                     (p.C0 * powf(2.0f, -(p.alpha + d / 2.0f) * j));
            
            int count_per_mask = 1;
            for (int i = 0; i < d; ++i) {
                 count_per_mask *= (h_k_ranges[j * d * 2 + i * 2 + 1] - h_k_ranges[j * d * 2 + i * 2] + 1);
            }

            int start_mask = (j == 0) ? 0 : 1;
            int end_mask = (j == 0) ? 1 : max_masks;
            for (int mask = start_mask; mask < end_mask; mask++) {
                int offset = h_offsets[j * max_masks + mask];
                
                int idx = threadIdx.x;
                while(idx < count_per_mask) {
                    int an_idx = offset + idx;
                    if(an_idx < total_coeffs) {
                        // The random numbers are in (0, 1]. Transform to (-1, 1].
                        h_coeffs[an_idx] = scale * (h_coeffs[an_idx] * 2.0f - 1.0f);
                    }
                    idx += blockDim.x;
                }
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
    const WaveletParams* __restrict__ params,
    const float* __restrict__ all_coeffs,
    const int* __restrict__ all_coeff_offsets,
    const int* __restrict__ all_k_ranges,
    int max_total_coeffs,
    int max_total_configs,
    float* __restrict__ Y_concat,
    const int* __restrict__ N_values,
    const int* __restrict__ offsets,
    int max_J)
{
    int func_idx = blockIdx.x;
    int n = threadIdx.x + blockIdx.y * blockDim.x;

    int N_current_func = N_values[func_idx];
    if (n >= N_current_func) return;

    WaveletParams p = params[func_idx];
    int d = p.d;
    int J = p.J;
    int max_masks = 1 << d;

    int x_offset = offsets[func_idx];
    int y_offset = offsets[func_idx];

    const float* coeffs = &all_coeffs[func_idx * max_total_coeffs];
    const int* coeff_offsets = &all_coeff_offsets[func_idx * max_total_configs];
    const int* k_ranges = &all_k_ranges[func_idx * (max_J + 1) * d * 2];
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
    const WaveletParams* h_params,
    int batch_size,
    const float* h_phi,
    const float* h_psi,
    const float* h_grid,
    int grid_len,
    int max_J,
    int max_N)
{
    int status = 0;
    float *d_X_concat = NULL, *d_Y_concat = NULL, *d_all_coeffs = NULL;
    int *d_all_coeff_offsets = NULL, *d_all_k_ranges = NULL, *d_total_coeffs_per_func = NULL;
    int *d_N_values = NULL, *d_offsets = NULL;
    WaveletParams* d_params = NULL;
    int* h_total_coeffs_per_func = NULL;
    curandGenerator_t* gens = NULL;
    int gens_created = 0;

    // Validate dimension to avoid overrunning fixed-size stack buffers in kernels
    if (d < 1 || d > MAX_DIM) {
        fprintf(stderr, "Invalid dimension d=%d (MAX_DIM=%d)\n", d, MAX_DIM);
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
    int max_count_per_mask = 1;
    int translations = 0;
    int max_total_coeffs = 0;
    dim3 grid_dim(1, 1);

    // 1. Copy wavelet data to constant memory
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

    // 2. Determine max buffer sizes
    max_masks = 1 << d;
    max_total_configs = (max_J + 1) * max_masks;
    translations = 1 << max_J;
    for (int i = 0; i < d; i++) {
        max_count_per_mask *= ((translations - 1) + 2 * K_side + 1);
    }
    max_total_coeffs = max_total_configs * max_count_per_mask;

    // 3. Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_X_concat, total_points * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y_concat, total_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_params, batch_size * sizeof(WaveletParams)));
    CUDA_CHECK(cudaMalloc(&d_all_coeffs, batch_size * max_total_coeffs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_coeff_offsets, batch_size * max_total_configs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_k_ranges, batch_size * (max_J + 1) * d * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total_coeffs_per_func, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_N_values, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, batch_size * sizeof(int)));

    // 4. Copy inputs from host to device
    CUDA_CHECK(cudaMemcpy(d_X_concat, h_X_concat, total_points * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params, h_params, batch_size * sizeof(WaveletParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N_values, h_N_values, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // 5. Compute metadata (offsets and counts)
    compute_metadata_and_scale_kernel<<<batch_size, THREADS_PER_BLOCK>>>(
        d_params, batch_size, max_total_coeffs, max_total_configs,
        d_all_coeffs, d_all_coeff_offsets, d_all_k_ranges, d_total_coeffs_per_func,
        max_J, true, false // compute metadata, do not scale
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 6. Copy total_coeffs back to host to know how many randoms to generate
    h_total_coeffs_per_func = (int*)malloc(batch_size * sizeof(int));
    if (h_total_coeffs_per_func == NULL) {
        fprintf(stderr, "Failed to allocate host memory for total_coeffs_per_func.\n");
        status = -1;
        goto cleanup;
    }
    CUDA_CHECK(cudaMemcpy(h_total_coeffs_per_func, d_total_coeffs_per_func, batch_size * sizeof(int), 
               cudaMemcpyDeviceToHost));

    // 7. Generate random numbers
    gens = (curandGenerator_t*)malloc(batch_size * sizeof(curandGenerator_t));
    if (gens == NULL) {
        fprintf(stderr, "Failed to allocate host memory for cuRAND generators.\n");
        status = -1;
        goto cleanup;
    }
    for (int i = 0; i < batch_size; i++) gens[i] = NULL;

    for (int func_idx = 0; func_idx < batch_size; func_idx++) {
        int total = h_total_coeffs_per_func[func_idx];
        if (total > max_total_coeffs) {
            fprintf(stderr, "Computed coefficient count %d exceeds buffer capacity %d\n", total, max_total_coeffs);
            status = -1;
            goto cleanup;
        }
        float* d_func_coeffs = d_all_coeffs + func_idx * max_total_coeffs;
        int aligned_total = (total % 2 == 0) ? total : total + 1; // curandGenerateUniform prefers even count
        if (aligned_total > max_total_coeffs) aligned_total = max_total_coeffs;
        
        CURAND_CHECK(curandCreateGenerator(&gens[func_idx], CURAND_RNG_PSEUDO_DEFAULT));
        gens_created++;
        int stream_idx = func_idx % g_num_streams;
        CURAND_CHECK(curandSetStream(gens[func_idx], g_streams[stream_idx]));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gens[func_idx], h_params[func_idx].seed));
        CURAND_CHECK(curandGenerateUniform(gens[func_idx], d_func_coeffs, aligned_total));
    }

    for (int i = 0; i < g_num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(g_streams[i]));
    }
    
    // 8. Scale the coefficients
    compute_metadata_and_scale_kernel<<<batch_size, THREADS_PER_BLOCK>>>(
        d_params, batch_size, max_total_coeffs, max_total_configs,
        d_all_coeffs, d_all_coeff_offsets, d_all_k_ranges, d_total_coeffs_per_func,
        max_J, false, true // do not compute metadata, scale coeffs
    );
    CUDA_CHECK(cudaPeekAtLastError());

    // 9. Evaluate all functions in the batch
    grid_dim = dim3(batch_size, (max_N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    evaluate_wavelet_batch_kernel<<<grid_dim, THREADS_PER_BLOCK>>>(
        d_X_concat, d_params, d_all_coeffs, d_all_coeff_offsets, d_all_k_ranges,
        max_total_coeffs, max_total_configs, d_Y_concat, d_N_values, d_offsets,
        max_J
    );
    CUDA_CHECK(cudaPeekAtLastError());

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
    if (h_total_coeffs_per_func != NULL) {
        free(h_total_coeffs_per_func);
    }
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
    free_device(d_params);
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
