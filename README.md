# Persistent Dense GEMM for NVIDIA Hopper with CuTeDSL 🔥

[![Releases](https://img.shields.io/badge/Release-Download-blue?logo=github&style=for-the-badge)](https://github.com/Hunter-XP/persistent_dense_gemm/releases)

![Hopper GPU](https://upload.wikimedia.org/wikipedia/commons/6/6f/NVIDIA_logo.svg)

Table of Contents
- About
- Key Features
- Why persistent GEMM on Hopper
- Requirements
- Quickstart
- Installation (download & execute)
- Build from source
- Usage examples
- Performance knobs
- Benchmark results
- API reference (core ideas)
- Testing
- Contributing
- License

About
This repository implements a persistent dense GEMM kernel tuned for NVIDIA Hopper-class GPUs. It uses CuTeDSL to express tiled tensor computations and to generate high-performance CUDA code. The project targets large-block, high-throughput matrix multiply workloads where persistence and register-block scheduling yield sustained performance.

Key Features
- Persistent-kernel design that reduces kernel launch overheads and maximizes SM occupancy.
- Dense GEMM microkernel written in CuTeDSL and lowered to CUDA for Hopper tensor cores.
- Multi-level tiling: register, shared, L2-friendly macro-tiles.
- Streaming prefetch into shared memory and L1.
- Support for mixed precision paths (FP64, FP32, TF32, FP16, BF16).
- Built-in autotuner for tile sizes and occupancy settings.
- Simple C++ host API to integrate into pipelines.

Why persistent GEMM on Hopper
Hopper GPUs expose large register files, ample shared memory, and improved tensor core throughput. A persistent-kernel approach keeps computation on the GPU and removes repeated kernel launch overheads for batched or streamed GEMM workloads. CuTeDSL gives a clear abstraction for tiling and lowering. This repo merges algorithmic tiling with hardware tuning to extract bandwidth and compute efficiency on Hopper.

Requirements
- Linux x86_64
- CUDA 11.8+ (Hopper benefits from newer drivers)
- NVIDIA driver compatible with your GPU
- CMake 3.18+
- A recent GCC or Clang supporting C++17
- CuTeDSL (submodule or install; see Build from source)
- Python 3.8+ (optional, for autotuner and scripts)

Quickstart
1. Visit the release page to get the tested runtime binary or installer:
   https://github.com/Hunter-XP/persistent_dense_gemm/releases
2. Download the release asset for your platform.
3. Run the installer or execute the binary supplied with the release.

Installation (download & execute)
Download the release asset from the project releases and execute the provided installer or binary. The releases page contains packaged builds and native installers. Example steps for a Linux installer:

1. Open the releases page:
   https://github.com/Hunter-XP/persistent_dense_gemm/releases

2. Download the appropriate asset, for example:
   - persistent_dense_gemm_hopper-linux-x86_64.tar.gz

3. Extract and run the installer:
```bash
# example commands; filenames vary by release
wget https://github.com/Hunter-XP/persistent_dense_gemm/releases/download/v1.0.0/persistent_dense_gemm_hopper-linux-x86_64.tar.gz
tar -xzf persistent_dense_gemm_hopper-linux-x86_64.tar.gz
cd persistent_dense_gemm_hopper
./install.sh
```

The installer will place libraries in /opt/persistent_dense_gemm or a local prefix. After install, you will have:
- libpersistent_gemm.so
- include/persistent_gemm.h
- examples/host_demo

If the release link does not work, check the Releases section on the repository page for assets and versioned installers.

Build from source
Clone the repo and build with CMake. The build system detects CuTeDSL location and the CUDA toolkit.

```bash
git clone https://github.com/Hunter-XP/persistent_dense_gemm.git
cd persistent_dense_gemm
git submodule update --init --recursive   # pulls CuTeDSL if configured as submodule
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
make -j$(nproc)
```

Key CMake flags
- PERSISTENT_GEMM_TUNER=ON to build the autotuner
- CUDA_ARCH=90 to target Hopper SM (set according to GPU)
- PREFIX=/your/install/path to change install location

Usage examples
Host API (C++ sketch)
```cpp
#include <persistent_gemm.h>

int main() {
  // allocate device buffers
  // A: MxK, B: KxN, C: MxN
  PersistentGemm::Config cfg;
  cfg.dtypeA = PersistentGemm::FP32;
  cfg.dtypeB = PersistentGemm::FP32;
  cfg.tileM = 256;
  cfg.tileN = 256;
  cfg.tileK = 32;
  cfg.persistent = true;
  auto gemm = PersistentGemm::create(cfg);
  gemm->run(A_device, B_device, C_device, M, N, K, stream);
  return 0;
}
```

Python wrapper example
```python
from persistent_gemm import Runner
r = Runner(dtype='tf32', tile_m=256, tile_n=256, tile_k=32)
r.run(A_ptr, B_ptr, C_ptr, M, N, K)
```

Performance knobs
- tileM / tileN: macro-tile shape across threadblocks.
- tileK: inner K tile moved per iteration.
- warps_per_block: controls thread-to-warp grouping.
- regs_per_warp: affects register blocking.
- prefetch_depth: number of K-steps prefetched to shared memory.
- persistence: enable to run a long-lived kernel which loops over input batches.

Tuning strategy
1. Start with moderate tiles (256x256x32).
2. Run roofline to identify memory or compute bound regime.
3. Adjust tileK to balance shared memory pressure and tensor core utilization.
4. Change warps_per_block to increase parallelism if occupancy is low.
5. Use the autotuner to sweep recommended ranges.

Autotuner
The repository includes a Python autotuner that runs microbenchmarks and selects good tile configurations. Invoke:
```bash
python tools/autotune.py --device 0 --problem 16384x16384x4096 --trials 50
```
The script writes the best config JSON to autotune_results/{device}_{timestamp}.json

Benchmark results
The project ships benchmark scripts that measure GEMM throughput (TFLOPS) across precisions. Typical behaviors on one Hopper SMX:
- TF32: near theoretical peak for large M/N/K with tileM=512, tileN=512, tileK=64.
- FP16/BF16: high efficiency when using mixed-precision accumulator paths.
- FP64: limited by tensor core absence (if applicable) and memory bandwidth.

Expect sustained throughput within 70-95% of peak compute on large matrices after autotune, depending on precision and PCIe/NVLink bandwidth.

Design overview
- CuTeDSL kernels express the tiled compute in a high-level schedule.
- Lowering passes translate to CUDA kernels with explicit shared memory loads and warp-level MMA calls.
- The persistent kernel keeps a block of threads bound to an assigned tile region and iterates over input batches or K-steps.
- Host-side driver queues work and signals termination. The host uses CUDA streams and events for overlap between upload and compute.

API reference (core ideas)
- PersistentGemm::Config: configure tile sizes, precision, persistence flag, and prefetch.
- PersistentGemm::create(cfg): allocate internal buffers and prepare kernels.
- runner->run(A,B,C,M,N,K,stream): enqueue persistent work; returns immediately for asynchronous runs.
- runner->sync(): block until all queued work completes.

Testing
Run unit tests and microbenchmarks:
```bash
cd build
ctest -j8
./bench/bench_gemm --size 8192 --dtype tf32
```
The test suite includes correctness checks versus cuBLAS on random matrices and deterministic seeds.

Contributing
- Fork the repository and open a feature branch.
- Run tests locally and ensure coding style matches clang-format.
- Write a short design note for non-trivial changes.
- Open a PR with a clear scope and a benchmark result for Hopper.

Issue guidelines
- Provide GPU model, driver version, CUDA version, and sample commands.
- Attach logs from the autotuner when performance differs.

Images and visual aids
- Architecture diagram: show tiling across registers, shared memory, and tensor cores.
- Performance plot: throughput vs problem size for TF32 and FP16.
(You will find these assets in the docs/ directory in releases and in the release assets linked above.)

Releases and downloads
Use the GitHub Releases page to get tested installers and prebuilt binaries:
https://github.com/Hunter-XP/persistent_dense_gemm/releases

Look for assets named for your OS and CUDA version. The release package contains:
- prebuilt libs
- example host apps
- autotuner scripts
- documentation PDF
Download the asset and execute the installer as shown in the Installation section.

License
This project uses an MIT-style license. See LICENSE.md in the repo for full terms.

Common pitfalls and troubleshooting
- Mismatched CUDA toolkit and driver cause runtime errors. Match the driver to CUDA requirements.
- Shared memory overcommit leads to kernel launch failure. Reduce tile sizes or increase partitioning.
- Autotuner may require multiple runs to stabilize results on multi-tenant systems.

Credits
- CuTeDSL for the high-level tensor scheduling primitives.
- NVIDIA Hopper architecture for the hardware target.
- Community contributions that tune microkernels and provide benchmarks.

Contact
Open issues on GitHub for bugs and feature requests. Use PRs to submit performance patches or new precision paths.