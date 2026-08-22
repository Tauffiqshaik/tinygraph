# TinyGraph — A Minimal Tensor Computation Graph with Custom CUDA Kernels

A small "compiler-adjacent" project: it represents a neural-net forward pass as an
**intermediate representation (IR)** — a graph of typed ops — and executes that graph
with **hand-written CUDA kernels**, including one **fused kernel** (bias-add + ReLU)
and a **shared-memory tiled matrix multiply**, benchmarked against naive versions.

This is intentionally small and honest: it is a learning project, not production
infrastructure. It exists to genuinely demonstrate — not just claim — familiarity with:
- Representing computation as a graph / IR (Input → MatMul → AddBias → ReLU)
- Writing and launching custom CUDA kernels
- GPU memory hierarchy: naive global-memory kernel vs. shared-memory tiled kernel
- Operator fusion: one fused kernel vs. two separate kernel launches
- Benchmarking and reasoning about the performance difference

## Architecture

```
include/graph.hpp     -> IR: Node, OpType, Graph (topological forward pass)
src/kernels.cu         -> CUDA kernels: naive matmul, tiled matmul, fused bias+relu
src/kernels.h           -> kernel launcher declarations
src/main.cu             -> builds a tiny 2-layer MLP graph, runs it, benchmarks
cpu_check/graph_cpu.cpp -> pure C++ CPU mirror of the same graph logic, used to
                           verify correctness without needing a GPU locally
```

## How to run it (no local GPU needed)

1. Open a new notebook at https://colab.research.google.com
2. Runtime → Change runtime type → **T4 GPU**
3. Upload this folder (or `git clone` your own repo once you push it to GitHub)
4. In a cell:
   ```
   !nvcc -O2 -o tinygraph src/main.cu src/kernels.cu -Iinclude
   !./tinygraph
   ```
5. It prints: graph structure, naive vs tiled matmul timing, fused vs unfused
   bias+relu timing, and a correctness check against the CPU reference.

## Suggested week plan

- **Day 1–2**: Get it compiling and running on Colab exactly as-is. Read every line —
  you should be able to explain what each kernel does line by line.
- **Day 3**: Modify `TILE_WIDTH` in `kernels.cu`, re-run, record how timing changes.
  This is your first real "performance experiment."
- **Day 4**: Add a second fused op of your own (e.g., fuse MatMul+AddBias into one
  kernel instead of two). This is genuinely extending the project, not copying it.
- **Day 5**: Write up a short RESULTS.md with your benchmark numbers and a 3–4
  sentence explanation of *why* the tiled/fused versions are faster (memory
  coalescing, shared memory reuse, fewer kernel launches / less global memory
  round-tripping).
- **Day 6–7**: Push to GitHub with a clear README (this one, edited to be yours),
  pin it on your profile, and add ONE honest bullet to your resume, e.g.:
  > "Built a small tensor computation graph in C++/CUDA with a fused bias+ReLU
  > kernel and shared-memory tiled matmul; benchmarked against naive kernels."

That's a real, defensible, interview-safe bullet — because you can open the repo
and explain every line if asked.

## What this does *not* claim

This is not compiler-passes, IRs at LLVM scale, distributed training, or
production kernel work. Don't oversell it on the resume — undersell slightly
and let the interviewer discover it's more thoughtful than expected, not less.
