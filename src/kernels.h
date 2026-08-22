#pragma once

// All matrices are row-major float arrays on the DEVICE.
// A: (M x K), B: (K x N), C: (M x N)

void matmulNaive(const float* dA, const float* dB, float* dC, int M, int K, int N);
void matmulTiled(const float* dA, const float* dB, float* dC, int M, int K, int N);

// bias: length N, broadcast across the M rows of A (M x N)
void addBias(const float* dA, const float* dBias, float* dC, int M, int N);
void reluInPlace(float* dC, int M, int N);

// Fused: C = max(A + bias, 0) in a single kernel launch (operator fusion)
void fusedBiasReLU(const float* dA, const float* dBias, float* dC, int M, int N);

// Returns elapsed milliseconds for a callable's device work using CUDA events.
// Defined as a template in kernels.h-adjacent .cu via explicit wrapper functions below.
float timeMatmulNaive(const float* dA, const float* dB, float* dC, int M, int K, int N, int iters);
float timeMatmulTiled(const float* dA, const float* dB, float* dC, int M, int K, int N, int iters);
float timeUnfusedBiasReLU(const float* dA, const float* dBias, float* dC, int M, int N, int iters);
float timeFusedBiasReLU(const float* dA, const float* dBias, float* dC, int M, int N, int iters);
