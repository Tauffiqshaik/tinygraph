#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "../include/graph.hpp"
#include "kernels.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static void fillRandom(std::vector<float>& v, unsigned seed) {
    srand(seed);
    for (auto& x : v) x = (float)(rand() % 200 - 100) / 100.0f; // [-1, 1]
}

// CPU reference matmul, used to sanity-check the GPU kernels agree with it.
static void matmulCPU(const std::vector<float>& A, const std::vector<float>& B,
                       std::vector<float>& C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

static float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    // -----------------------------------------------------------------
    // 1. Build the IR: a tiny 2-layer MLP forward pass
    //    x -> MatMul(W1) -> FusedBiasReLU(b1) -> MatMul(W2) -> AddBias(b2)
    // -----------------------------------------------------------------
    Graph g;
    int x  = g.addInput("x",  {64, 128});
    int w1 = g.addInput("W1", {128, 256});
    int b1 = g.addInput("b1", {1, 256});
    int w2 = g.addInput("W2", {256, 10});
    int b2 = g.addInput("b2", {1, 10});

    int h1 = g.addMatMul(x, w1, "h1 = x @ W1");
    int a1 = g.addFusedBiasReLU(h1, b1, "a1 = relu(h1 + b1)   [fused kernel]");
    int h2 = g.addMatMul(a1, w2, "h2 = a1 @ W2");
    int out = g.addBias(h2, b2, "out = h2 + b2");

    printf("=== TinyGraph IR ===\n");
    g.print();
    printf("\n");

    // -----------------------------------------------------------------
    // 2. Correctness check: naive kernel vs tiled kernel vs CPU reference
    //    on the first MatMul in the graph (x @ W1)
    // -----------------------------------------------------------------
    int M = 64, K = 128, N = 256;
    std::vector<float> hA(M * K), hB(K * N), hC_naive(M * N), hC_tiled(M * N), hC_cpu(M * N);
    fillRandom(hA, 1);
    fillRandom(hB, 2);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    matmulNaive(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaMemcpy(hC_naive.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    matmulTiled(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaMemcpy(hC_tiled.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    matmulCPU(hA, hB, hC_cpu, M, K, N);

    printf("=== Correctness check (max abs diff vs CPU reference) ===\n");
    printf("naive kernel : %.6f\n", maxAbsDiff(hC_naive, hC_cpu));
    printf("tiled kernel : %.6f\n", maxAbsDiff(hC_tiled, hC_cpu));
    printf("(should both be ~1e-4 or smaller — float rounding only)\n\n");

    // -----------------------------------------------------------------
    // 3. Benchmark: naive vs tiled matmul at a larger, more realistic size
    // -----------------------------------------------------------------
    int bM = 512, bK = 512, bN = 512;
    std::vector<float> big_hA(bM * bK), big_hB(bK * bN);
    fillRandom(big_hA, 3);
    fillRandom(big_hB, 4);

    float *bdA, *bdB, *bdC;
    CUDA_CHECK(cudaMalloc(&bdA, bM * bK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bdB, bK * bN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bdC, bM * bN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(bdA, big_hA.data(), bM * bK * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bdB, big_hB.data(), bK * bN * sizeof(float), cudaMemcpyHostToDevice));

    float tNaive = timeMatmulNaive(bdA, bdB, bdC, bM, bK, bN, 20);
    float tTiled = timeMatmulTiled(bdA, bdB, bdC, bM, bK, bN, 20);

    printf("=== Benchmark: %dx%d @ %dx%d matmul (avg of 20 runs) ===\n", bM, bK, bK, bN);
    printf("naive (global memory) : %.4f ms\n", tNaive);
    printf("tiled (shared memory) : %.4f ms\n", tTiled);
    printf("speedup                : %.2fx\n\n", tNaive / tTiled);

    // -----------------------------------------------------------------
    // 4. Benchmark: fused bias+relu vs two separate kernel launches
    // -----------------------------------------------------------------
    int fM = 2048, fN = 2048;
    std::vector<float> f_hA(fM * fN), f_hBias(fN);
    fillRandom(f_hA, 5);
    fillRandom(f_hBias, 6);

    float *fdA, *fdBias, *fdC;
    CUDA_CHECK(cudaMalloc(&fdA, fM * fN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fdBias, fN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fdC, fM * fN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(fdA, f_hA.data(), fM * fN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(fdBias, f_hBias.data(), fN * sizeof(float), cudaMemcpyHostToDevice));

    float tUnfused = timeUnfusedBiasReLU(fdA, fdBias, fdC, fM, fN, 50);
    float tFused   = timeFusedBiasReLU(fdA, fdBias, fdC, fM, fN, 50);

    printf("=== Benchmark: %dx%d bias+relu (avg of 50 runs) ===\n", fM, fN);
    printf("unfused (2 kernel launches) : %.4f ms\n", tUnfused);
    printf("fused   (1 kernel launch)   : %.4f ms\n", tFused);
    printf("speedup                      : %.2fx\n", tUnfused / tFused);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(bdA); cudaFree(bdB); cudaFree(bdC);
    cudaFree(fdA); cudaFree(fdBias); cudaFree(fdC);
    return 0;
}
