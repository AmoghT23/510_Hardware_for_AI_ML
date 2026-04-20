%%writefile gemm_tiled.cu
// gemm_tiled.cu
// 1024x1024 FP32 matrix multiply -- shared-memory tiled kernel, TILE=8.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N     1024
#define TILE  8       // tile size as required (8x8 = 64 threads/block)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    int numTiles = n / TILE;

    for (int t = 0; t < numTiles; ++t) {
        As[ty][tx] = A[row * n + (t * TILE + tx)];
        Bs[ty][tx] = B[(t * TILE + ty) * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
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

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    gemm_tiled<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 20;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; ++i) {
        gemm_tiled<<<grid, block>>>(dA, dB, dC, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    float avg_ms  = ms_total / RUNS;
    double flops  = 2.0 * (double)N * N * N;
    double gflops = flops / (avg_ms * 1.0e6);
    double gbs    = (3.0 * bytes) / (avg_ms * 1.0e6);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    cpu_matmul(hA, hB, hRef, N);
    double max_err = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double e = fabs((double)hC[i] - (double)hRef[i]);
        if (e > max_err) max_err = e;
    }

    printf("\n");
    printf("+------------------------------------------------------------+\n");
    printf("|         TILED GEMM  TILE=%d  (1024 x 1024 FP32)             |\n", TILE);
    printf("+------------------------------------------------------------+\n");
    printf("| Block size      : %4d x %-4d  (%5d threads/block)        |\n",
           TILE, TILE, TILE*TILE);
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
