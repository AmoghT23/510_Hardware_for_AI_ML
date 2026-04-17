%%writefile gemm_naive.cu
// gemm_naive.cu
// 1024x1024 FP32 matrix multiply -- naive kernel (one thread per output element).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N      1024
#define BLOCK  16     // 16x16 = 256 threads per block

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            acc += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = acc;
    }
}

static void cpu_matmul(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
    }
}

int main() {
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *hA   = (float*)malloc(bytes);
    float *hB   = (float*)malloc(bytes);
    float *hC   = (float*)malloc(bytes);
    float *hRef = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * N; ++i) {
        hA[i] = (float)rand() / RAND_MAX;
        hB[i] = (float)rand() / RAND_MAX;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    // Warm-up
    gemm_naive<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 20;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; ++i) {
        gemm_naive<<<grid, block>>>(dA, dB, dC, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    float avg_ms  = ms_total / RUNS;
    double flops  = 2.0 * (double)N * N * N;
    double gflops = flops / (avg_ms * 1.0e6);
    double gbs    = (3.0 * bytes) / (avg_ms * 1.0e6);  // rough: A+B read, C write

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    cpu_matmul(hA, hB, hRef, N);
    double max_err = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double e = fabs((double)hC[i] - (double)hRef[i]);
        if (e > max_err) max_err = e;
    }

    printf("\n");
    printf("+------------------------------------------------------------+\n");
    printf("|              NAIVE GEMM  (1024 x 1024 FP32)                |\n");
    printf("+------------------------------------------------------------+\n");
    printf("| Block size      : %4d x %-4d  (%5d threads/block)        |\n",
           BLOCK, BLOCK, BLOCK*BLOCK);
    printf("| Grid size       : %4d x %-4d                              |\n",
           grid.x, grid.y);
    printf("| Iterations      : %-4d                                     |\n", RUNS);
    printf("|------------------------------------------------------------|\n");
    printf("| Avg time        : %10.3f ms                            |\n", avg_ms);
    printf("| Throughput      : %10.2f GFLOP/s                       |\n", gflops);
    printf("| Effective BW    : %10.2f GB/s   (naive estimate)       |\n", gbs);
    printf("|------------------------------------------------------------|\n");
    printf("| Max abs error   : %10.6f          [%s]                |\n",
           max_err, max_err < 1e-2 ? " OK " : "FAIL");
    printf("+------------------------------------------------------------+\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}

//Compile with nvcc
!nvcc -O3 -arch=sm_75 gemm_naive.cu -o gemm_naive

!./gemm_naive
