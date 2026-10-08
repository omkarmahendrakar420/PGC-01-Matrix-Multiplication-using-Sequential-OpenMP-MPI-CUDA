# Parallel & GPU Computing

> **Course:** Parallel & GPU Computing (PGC)  
> **Student Name:** OMKAR S MAHENDRAKAR  
> **USN:** `01FE25BCI703`  

This repository contains practical implementations, source code, and experimental results for the **Parallel & GPU Computing** course. It focuses on analyzing and comparing the computational performance of dense matrix multiplication across different execution paradigms: single-core sequential CPU baseline, multi-core OpenMP shared memory, distributed-memory OpenMPI clusters, and massively parallel NVIDIA CUDA GPU acceleration.

---

## Experiments

### Experiment 1: Dense Matrix Multiplication Benchmarks ($4000 \times 4000$)

| Part | Experiment | Implementation Paradigm | Documentation |
| :--- | :--- | :--- | :--- |
| **Part A** | Matrix Multiplication | Sequential (Baseline) | [Part A - Sequential README](Experiment-1/Part-A-Sequential/README.md) |
| **Part B** | Matrix Multiplication | OpenMP (Shared Memory Multi-threading) | [Part B - OpenMP README](Experiment-1/Part-B-OpenMP/README.md) |
| **Part C** | Matrix Multiplication | OpenMPI (Distributed-Memory Cluster) | [Part C - MPI README](Experiment-1/Part-C-MPI/README.md) |
| **Part D** | Matrix Multiplication | NVIDIA CUDA (Massively Parallel GPU) | [Part D - CUDA README](Experiment-1/Part-D-CUDA/README.md) |

---

## Benchmark Performance Summary Matrix

| Implementation Paradigm | Compute Architecture | Execution Time | Speedup ($S$) | Parallel Efficiency ($E$) | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential CPU** | 1 Core (Single-Threaded Baseline) | **244.120000 s** | **1.00×** (Ref) | 100.0% | `C[0][0] = 4000.00` (PASS) |
| **MPI Cluster** | 4 Distributed Nodes | **92.979510 s** | **2.63×** | 65.8% | `C[0][0] = 4000.00` (PASS) |
| **OpenMP Multi-Core** | 8 vCPU Cores (Shared Memory) | **40.545825 s** | **6.02×** | **75.3%** | `C[0][0] = 4000.00` (PASS) |
| **NVIDIA CUDA (Total)** | 16,000,000 GPU Threads (PCIe DMA + Kernel) | **0.343020 s** | **711.68×** | — | `C[0][0] = 4000.00` (PASS) |
| **NVIDIA CUDA (Kernel)** | 16,000,000 GPU Threads (Pure Compute) | **0.316872 s** | **770.40×** | — | `C[0][0] = 4000.00` (PASS) |

---

## Repository Structure

```text
Parallel-GPU-Computing/
│
├── README.md
│
└── Experiment-1/
    ├── Part-A-Sequential/
    │   ├── README.md
    │   ├── code/
    │   │   └── sequential_matrix.c
    │   └── screenshots/
    │       ├── Sequential Matrix Output 1.png
    │       └── ... (9 images)
    │
    ├── Part-B-OpenMP/
    │   ├── README.md
    │   ├── code/
    │   │   └── openmp_matrix.c
    │   └── screenshots/
    │       ├── OpenMP Matrix Output 1.png
    │       └── ... (5 images)
    │
    ├── Part-C-MPI/
    │   ├── README.md
    │   ├── code/
    │   │   └── matrix_mpi.c
    │   └── screenshots/
    │       └── ... (21 images)
    │
    └── Part-D-CUDA/
        ├── README.md
        ├── code/
        │   └── matrix_cuda.cu
        └── screenshots/
            └── CUDA Matrix Output 1.png
```

---

## Technologies Used

* **C / C++** (Standard C99 / C11 / CUDA C++)
* **OpenMP** (Open Multi-Processing API)
* **OpenMPI** (Message Passing Interface standard)
* **NVIDIA CUDA** (Compute Unified Device Architecture)
* **GCC / NVCC** (GNU Compiler Collection & NVIDIA CUDA Compiler)
