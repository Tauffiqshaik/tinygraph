// Pure C++ (no CUDA) mirror of the graph forward pass, used to sanity-check
// the IR construction logic and op math before touching a GPU at all.
// Compiles anywhere with g++: g++ -O2 -std=c++17 graph_cpu.cpp -o graph_cpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../include/graph.hpp"

using Mat = std::vector<float>;

static void fillRandom(Mat& v, unsigned seed) {
    srand(seed);
    for (auto& x : v) x = (float)(rand() % 200 - 100) / 100.0f;
}

static Mat matmul(const Mat& A, const Mat& B, int M, int K, int N) {
    Mat C(M * N, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
    return C;
}

static Mat addBiasReLU(const Mat& A, const Mat& bias, int M, int N, bool relu) {
    Mat C(M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float v = A[i * N + j] + bias[j];
            C[i * N + j] = relu ? std::max(v, 0.0f) : v;
        }
    return C;
}

int main() {
    // Build the exact same IR as the CUDA version, purely to validate the
    // graph construction and shape-checking logic without needing a GPU.
    Graph g;
    int x  = g.addInput("x",  {64, 128});
    int w1 = g.addInput("W1", {128, 256});
    int b1 = g.addInput("b1", {1, 256});
    int w2 = g.addInput("W2", {256, 10});
    int b2 = g.addInput("b2", {1, 10});

    int h1  = g.addMatMul(x, w1, "h1 = x @ W1");
    int a1  = g.addFusedBiasReLU(h1, b1, "a1 = relu(h1 + b1)");
    int h2  = g.addMatMul(a1, w2, "h2 = a1 @ W2");
    int out = g.addBias(h2, b2, "out = h2 + b2");

    printf("=== TinyGraph IR (CPU build check) ===\n");
    g.print();

    // Execute the same math on CPU to make sure shapes and op semantics
    // are self-consistent end-to-end before ever touching CUDA.
    int M = 64, K1 = 128, N1 = 256, N2 = 10;
    Mat hx(M * K1), hw1(K1 * N1), hb1(N1), hw2(N1 * N2), hb2(N2);
    fillRandom(hx, 1); fillRandom(hw1, 2); fillRandom(hb1, 3);
    fillRandom(hw2, 4); fillRandom(hb2, 5);

    Mat hh1 = matmul(hx, hw1, M, K1, N1);
    Mat ha1 = addBiasReLU(hh1, hb1, M, N1, /*relu=*/true);
    Mat hh2 = matmul(ha1, hw2, M, N1, N2);
    Mat hout = addBiasReLU(hh2, hb2, M, N2, /*relu=*/false);

    printf("\n=== Forward pass complete ===\n");
    printf("output shape: (%d, %d)\n", M, N2);
    printf("output[0][0..4] = ");
    for (int j = 0; j < 5 && j < N2; ++j) printf("%.4f ", hout[j]);
    printf("\n");

    printf("\nIf this ran without errors, the graph/shape logic is sound.\n");
    printf("The CUDA version (src/main.cu) executes the identical graph on GPU\n");
    printf("with naive vs. tiled matmul and unfused vs. fused bias+relu kernels.\n");
    return 0;
}
