#include "kernels.h"
#include <cuda_runtime.h>
#include <algorithm>

#define TILE_WIDTH 16

// ---------------------------------------------------------------------------
// Naive matmul: each thread computes one output element, reading directly
// from global memory K times. Simple, but memory-bandwidth bound — every
// thread re-reads the same rows/columns of A and B that its neighbors do.
// ---------------------------------------------------------------------------
__global__ void matmulNaiveKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Tiled matmul: each thread block cooperatively loads TILE_WIDTH x TILE_WIDTH
// tiles of A and B into shared memory, so each value loaded from global
// memory is reused TILE_WIDTH times by threads in the block instead of being
// re-fetched from global memory by every thread individually. This is the
// standard first step in optimizing GPU matmul (before Tensor Cores / cuBLAS).
// ---------------------------------------------------------------------------
__global__ void matmulTiledKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Unfused bias-add and ReLU: two separate kernels, two separate passes over
// global memory for the full (M x N) tensor.
// ---------------------------------------------------------------------------
__global__ void addBiasKernel(const float* A, const float* bias, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    C[row * N + col] = A[row * N + col] + bias[col];
}

__global__ void reluKernel(float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    C[row * N + col] = fmaxf(C[row * N + col], 0.0f);
}

// ---------------------------------------------------------------------------
// FUSED kernel: bias-add + ReLU in one pass. One kernel launch, one read of
// A and bias, one write of C — instead of two launches and an extra
// round-trip of the intermediate result through global memory.
// This is the "operator fusion" the JD refers to, at toy scale.
// ---------------------------------------------------------------------------
__global__ void fusedBiasReLUKernel(const float* A, const float* bias, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    C[row * N + col] = fmaxf(A[row * N + col] + bias[col], 0.0f);
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
static dim3 grid2D(int M, int N, dim3 block) {
    return dim3((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
}

void matmulNaive(const float* dA, const float* dB, float* dC, int M, int K, int N) {
    dim3 block(16, 16);
    dim3 grid = grid2D(M, N, block);
    matmulNaiveKernel<<<grid, block>>>(dA, dB, dC, M, K, N);
}

void matmulTiled(const float* dA, const float* dB, float* dC, int M, int K, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid = grid2D(M, N, block);
    matmulTiledKernel<<<grid, block>>>(dA, dB, dC, M, K, N);
}

void addBias(const float* dA, const float* dBias, float* dC, int M, int N) {
    dim3 block(16, 16);
    dim3 grid = grid2D(M, N, block);
    addBiasKernel<<<grid, block>>>(dA, dBias, dC, M, N);
}

void reluInPlace(float* dC, int M, int N) {
    dim3 block(16, 16);
    dim3 grid = grid2D(M, N, block);
    reluKernel<<<grid, block>>>(dC, M, N);
}

void fusedBiasReLU(const float* dA, const float* dBias, float* dC, int M, int N) {
    dim3 block(16, 16);
    dim3 grid = grid2D(M, N, block);
    fusedBiasReLUKernel<<<grid, block>>>(dA, dBias, dC, M, N);
}

// ---------------------------------------------------------------------------
// Timing helpers using CUDA events (device-side timing, not wall clock)
// ---------------------------------------------------------------------------
float timeMatmulNaive(const float* dA, const float* dB, float* dC, int M, int K, int N, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warmup
    matmulNaive(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) matmulNaive(dA, dB, dC, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

float timeMatmulTiled(const float* dA, const float* dB, float* dC, int M, int K, int N, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    matmulTiled(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) matmulTiled(dA, dB, dC, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

float timeUnfusedBiasReLU(const float* dA, const float* dBias, float* dC, int M, int N, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    addBias(dA, dBias, dC, M, N);
    reluInPlace(dC, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        addBias(dA, dBias, dC, M, N);
        reluInPlace(dC, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

float timeFusedBiasReLU(const float* dA, const float* dBias, float* dC, int M, int N, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    fusedBiasReLU(dA, dBias, dC, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) fusedBiasReLU(dA, dBias, dC, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}
