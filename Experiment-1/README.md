# Experiment 1: Dense Matrix Multiplication Benchmarks

> **Student Name:** OMKAR S MAHENDRAKAR  
> **USN:** `01FE25BCI703`  
> **Course:** Parallel & GPU Computing (PGC)  

---

## Overview

This experiment evaluates the performance, scaling characteristics, and hardware efficiency of dense matrix multiplication ($4000 \times 4000$, 128 GFLOPs) across four computational paradigms:

1. **Part A: Sequential Baseline** — Single-core CPU computation.
2. **Part B: OpenMP** — Shared-memory multi-threading with 8 CPU threads.
3. **Part C: MPI** — Distributed-memory cluster computation across 4 VM nodes.
4. **Part D: CUDA** — Massively parallel GPU acceleration with 16,000,000 threads.

---

## Directory Navigation

| Section | Description | Source Code | Documentation |
| :--- | :--- | :--- | :--- |
| **Part A** | Sequential Matrix Multiplication | [sequential_matrix.c](Part-A-Sequential/code/sequential_matrix.c) | [Part A Documentation](Part-A-Sequential/README.md) |
| **Part B** | OpenMP Matrix Multiplication | [openmp_matrix.c](Part-B-OpenMP/code/openmp_matrix.c) | [Part B Documentation](Part-B-OpenMP/README.md) |
| **Part C** | OpenMPI Distributed Cluster | [matrix_mpi.c](Part-C-MPI/code/matrix_mpi.c) | [Part C Documentation](Part-C-MPI/README.md) |
| **Part D** | NVIDIA CUDA GPU Acceleration | [matrix_cuda.cu](Part-D-CUDA/code/matrix_cuda.cu) | [Part D Documentation](Part-D-CUDA/README.md) |

---

## Performance Summary Table

| Implementation | Hardware Architecture | Execution Time | Speedup ($S$) | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential CPU** | 1 Core (Single-Threaded Baseline) | `244.120000 s` | **1.00×** (Ref) | `C[0][0] = 4000.00` |
| **MPI Cluster** | 4 Distributed Cluster Nodes | `92.979510 s` | **2.63×** | `C[0][0] = 4000.00` |
| **OpenMP Multi-Core** | 8 vCPU Cores (Shared Memory) | `40.545825 s` | **6.02×** | `C[0][0] = 4000.00` |
| **NVIDIA CUDA (Total)** | 16,000,000 GPU Threads (PCIe DMA) | `0.343020 s` | **711.68×** | `C[0][0] = 4000.00` |
| **NVIDIA CUDA (Kernel)** | 16,000,000 GPU Threads (Pure Compute) | **`0.316872 s`** | **770.40×** | `C[0][0] = 4000.00` |
