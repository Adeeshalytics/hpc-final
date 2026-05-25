**HPC Final Project — Parallel Matrix Multiplication**

**Overview**
- **Summary:** This repository implements and benchmarks matrix multiplication (C = A×B) using five approaches: serial, OpenMP, Pthreads, MPI, and a Hybrid MPI+OpenMP model. The goal is to compare correctness, performance, and scaling behavior for square matrices of varying sizes.
- **Motivation:** Matrix multiplication is at the core of neural network computations; optimizing it with parallel techniques improves throughput for modern ML workloads.

**Quick Start**
- **Build:** `make`
- **Run benchmarks:** `python scripts/run_benchmarks.py`
- **View results:** open the `outputs/` and `results/` folders

**Highlights**
- All parallel variants reproduce the serial result exactly (RMSE = 0 for integer-valued matrices used in testing).
- The repository contains well-instrumented timing to produce speedup, efficiency, and GFLOPS measurements used in the analysis.

**Requirements**
- Compiler: GCC (MSYS2 / MinGW UCRT64 recommended)
- MPI library: Microsoft MPI (MS-MPI) v10 (for MPI runs)
- Python 3 (for benchmark orchestration and analysis)
- Optional: `make` to build via the provided `Makefile`

**Build & Run (Windows / MSYS2 example)**
1. Build the C implementations:

```bash
make
```

2. Run the benchmark harness (runs all variants and collects outputs):

```bash
python scripts/run_benchmarks.py
```

3. Generated outputs and CSVs are placed in the `outputs/` and `results/` folders.

**Reproducing the Paper Results**
- The experiments in the accompanying project report were run on an 8-core AMD Ryzen 7 8840HS (16 logical threads) with 16GB RAM. Results include strong-scaling (N=1024), problem-size sweeps (N from 256 to 2048), hybrid configuration sweeps, and MPI phase breakdowns.
- To reproduce, run the benchmark script with the same runtime parameters (matrix sizes, repeat counts). See `scripts/run_benchmarks.py` for the exact parameters used in the paper.

**Project Structure**
- Source implementations: [src/](src/)
- Benchmarking & analysis scripts: [scripts/](scripts/)
- Raw outputs: [outputs/](outputs/)
- Processed results / CSVs: [results/](results/)
- Figures used in the report are in: [figures/](figures/)

**Key Files**
- `src/serial_matrix.c`, `src/openmp_matrix.c`, `src/pthreads_matrix.c`, `src/mpi_matrix.c`, `src/hybrid_matrix.c` — core implementations.
- `scripts/run_benchmarks.py` — runs the experiments and writes `outputs/`.
- `scripts/analyze_benchmarks.py` and `scripts/generate_charts.py` — produce the charts and CSV tables used in the writeup.

**Validation & Correctness**
- Each implementation writes its output matrix to a file which is compared against the serial output using a Python validation script (RMSE, max error). With the integer-valued test matrices used, all parallel variants matched the serial result exactly.

**Authors & Credits**
- Group 07 — Adeesha M.G.P. (EG/2021/4385), A.N. Akarshana (EG/2021/4392), K. Akshayan (EG/2021/4393)

**Where to look first**
- Start with [src/serial_matrix.c](src/serial_matrix.c) to understand the baseline implementation.
- Inspect [scripts/run_benchmarks.py](scripts/run_benchmarks.py) to see how experiments are configured and invoked.
- Open `results/bench_hybrid.csv` and other CSVs to inspect numeric results quickly.

**Tips**
- On Windows use the provided `Makefile` with MSYS2 UCRT64 GCC and MS-MPI installed.
- For MPI runs, use the MS-MPI launcher (or `mpiexec`) as configured in the environment.

**License**
- No license file included. Add a `LICENSE` if you wish to make this project publicly reusable.

If you'd like, I can also:
- add quick badges (build / license),
- extract and embed a short results table and one figure into this README, or
- add a small `RUNNING.md` with step-by-step reproduction commands.
