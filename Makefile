# Build the CUDA version (needs nvcc — e.g. on Google Colab with a GPU runtime,
# or any machine with the CUDA toolkit installed).
tinygraph: src/main.cu src/kernels.cu
	nvcc -O2 -Iinclude -o tinygraph src/main.cu src/kernels.cu

# Build the CPU-only reference (no GPU/CUDA needed — works anywhere with g++).
cpu_check: cpu_check/graph_cpu.cpp
	g++ -O2 -std=c++17 -Iinclude cpu_check/graph_cpu.cpp -o cpu_check/graph_cpu

clean:
	rm -f tinygraph cpu_check/graph_cpu

.PHONY: clean
