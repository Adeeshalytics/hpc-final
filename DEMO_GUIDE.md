# Group Demo Guide — Individual Variant Commands

This guide gives **each group member** the exact PowerShell commands to demonstrate their variant. Designed for sequential demo from a single machine (one terminal, one person at a time).

---

## Pre-demo setup (run ONCE, by anyone)

Before any demo starts:

```powershell
cd C:\Projects\HPC
mingw32-make clean
mingw32-make all
```

This builds all 5 executables into `bin/`. Verify with:

```powershell
ls bin
```

You should see: `serial_gemm.exe`, `openmp_matrix.exe`, `pthreads_matrix.exe`, `mpi_matrix.exe`, `hybrid_matrix.exe`.

> **Important PowerShell rules** to remember during the demo:
> - To set an environment variable in PowerShell: `$env:OMP_NUM_THREADS = "8"` (NOT `OMP_NUM_THREADS=8` like bash)
> - To run a quoted path: prefix with `&` — e.g. `& "C:/MPI/Bin/mpiexec.exe"` (NOT just `"C:/MPI/Bin/mpiexec.exe"`)
> - Plain `./bin/serial_gemm.exe` works fine since there's no space in the path

---

## Variant 1 — Serial (Baseline)

> Shows the un-parallelised baseline. All other variants are measured against this.

### Build (only this variant)
```powershell
mingw32-make serial
```

### Run with default N=1024
```powershell
./bin/serial_gemm.exe 1024
```

### Run with different problem sizes
```powershell
./bin/serial_gemm.exe 256
./bin/serial_gemm.exe 512
./bin/serial_gemm.exe 1024
./bin/serial_gemm.exe 2048
```

### What to point out during the demo
- "No parallelism — one thread does all `2·N³` operations sequentially."
- "The `i-k-j` loop order gives cache-friendly access to B — a 12× speedup over the textbook order."
- Notice GFLOPS **drops** as N grows past 1024 (memory wall — the working set exceeds L3 cache).

### Expected output pattern
```
Serial GEMM | N=1024
Time: 0.XXX s | Performance: X.XX GFLOPS
```

---

## Variant 2 — OpenMP (Shared Memory, Pragmas)

> Shows the simplest shared-memory parallelism — one compiler directive.

### Build (only this variant)
```powershell
mingw32-make openmp
```

### Run with 1 thread (degenerate — should be slower than serial!)
```powershell
$env:OMP_NUM_THREADS = "1"
./bin/openmp_matrix.exe 1024
```

### Run with 2, 4, 8, 16 threads (strong scaling demo)
```powershell
$env:OMP_NUM_THREADS = "2";  ./bin/openmp_matrix.exe 1024
$env:OMP_NUM_THREADS = "4";  ./bin/openmp_matrix.exe 1024
$env:OMP_NUM_THREADS = "8";  ./bin/openmp_matrix.exe 1024
$env:OMP_NUM_THREADS = "16"; ./bin/openmp_matrix.exe 1024
```

### What to point out during the demo
- "The entire parallelisation is a single `#pragma omp parallel for` — look at the source."
- "1 thread is **slower** than serial — OpenMP runtime overhead is unconditional."
- "Speedup saturates around 8 threads (physical core count); 16 threads (SMT) gives no further gain."

### Expected output pattern
```
OpenMP GEMM | N=1024 | threads=8
Time: 0.XXX s | Performance: XX.XX GFLOPS
```

### Show the source (highlight the single magic line)
```powershell
Get-Content src/openmp_matrix.c | Select-String -Pattern "pragma" -Context 0,5
```

---

## Variant 3 — Pthreads (Shared Memory, Manual)

> Shows what OpenMP hides — manual thread creation, work assignment, joining.

### Build (only this variant)
```powershell
mingw32-make pthreads
```

### Run with N=1024, varying thread count (last arg = thread count)
```powershell
./bin/pthreads_matrix.exe 1024 1
./bin/pthreads_matrix.exe 1024 2
./bin/pthreads_matrix.exe 1024 4
./bin/pthreads_matrix.exe 1024 8
./bin/pthreads_matrix.exe 1024 16
```

### Run with different problem sizes (8 threads)
```powershell
./bin/pthreads_matrix.exe 256  8
./bin/pthreads_matrix.exe 512  8
./bin/pthreads_matrix.exe 1024 8
./bin/pthreads_matrix.exe 2048 8
```

### What to point out during the demo
- "Notice the source is ~30 lines longer than OpenMP — `pthread_create` / `pthread_join`, manual row assignment per thread, a `thread_arg_t` struct."
- "Performance is essentially identical to OpenMP — both call the same OS-level threading primitives."
- "Pthreads is what you'd use in C without OpenMP support — it's lower-level and more portable."

### Expected output pattern
```
Pthreads GEMM | N=1024 | threads=8
Time: 0.XXX s | Performance: XX.XX GFLOPS
```

### Show the source (highlight the worker function)
```powershell
Get-Content src/pthreads_matrix.c | Select-String -Pattern "pthread_create|worker|thread_arg" -Context 0,2
```

---

## Variant 4 — MPI (Distributed Memory)

> Shows distributed-memory message passing — the model that scales across machines.

### Build (only this variant)
```powershell
mingw32-make mpi
```

### Run with 1 process (no communication, like serial)
```powershell
& "C:/MPI/Bin/mpiexec.exe" -n 1 ./bin/mpi_matrix.exe 1024
```

### Run with 2, 4, 8, 16 processes (strong scaling demo)
```powershell
& "C:/MPI/Bin/mpiexec.exe" -n 2  ./bin/mpi_matrix.exe 1024
& "C:/MPI/Bin/mpiexec.exe" -n 4  ./bin/mpi_matrix.exe 1024
& "C:/MPI/Bin/mpiexec.exe" -n 8  ./bin/mpi_matrix.exe 1024
& "C:/MPI/Bin/mpiexec.exe" -n 16 ./bin/mpi_matrix.exe 1024
```

### What to point out during the demo
- "Output shows a **phase breakdown** — Bcast B, Scatter A, Compute, Gather C."
- "At 1 process: 98 % compute, 2 % comm. At 16 processes: **29 % compute, 71 % comm.**"
- "This is exactly when adding more processes starts to **hurt** performance — the comm overhead exceeds the compute savings."
- "Unlike OpenMP, MPI works across machines (a cluster) — but pays the cost of explicit message passing."

### Expected output pattern
```
MPI GEMM | N=1024 | processes=8
Total: 0.XXX s | Performance: XX.XX GFLOPS
Phase breakdown (per-phase max across ranks; % of phase-sum):
  Broadcast B : 0.XXX s ( X.X%)
  Scatter   A : 0.XXX s ( X.X%)
  Compute     : 0.XXX s ( X.X%)
  Gather    C : 0.XXX s ( X.X%)
  Comm total  : 0.XXX s ( X.X%)
```

### Show the source (highlight the collectives)
```powershell
Get-Content src/mpi_matrix.c | Select-String -Pattern "MPI_Bcast|MPI_Scatterv|MPI_Gatherv" -Context 0,1
```

---

## Variant 5 — Hybrid (MPI + OpenMP)

> The most general — MPI between processes (across nodes), OpenMP threads within each process.

### Build (only this variant)
```powershell
mingw32-make hybrid
```

### Run with 2 procs × 4 threads (= 8 total units)
```powershell
$env:OMP_NUM_THREADS = "4"
& "C:/MPI/Bin/mpiexec.exe" -n 2 ./bin/hybrid_matrix.exe 1024
```

### Run several hybrid configurations totalling 8 or 16 units
```powershell
# 8-unit configs (matching physical core count)
$env:OMP_NUM_THREADS = "8"; & "C:/MPI/Bin/mpiexec.exe" -n 1 ./bin/hybrid_matrix.exe 1024
$env:OMP_NUM_THREADS = "4"; & "C:/MPI/Bin/mpiexec.exe" -n 2 ./bin/hybrid_matrix.exe 1024
$env:OMP_NUM_THREADS = "2"; & "C:/MPI/Bin/mpiexec.exe" -n 4 ./bin/hybrid_matrix.exe 1024
$env:OMP_NUM_THREADS = "1"; & "C:/MPI/Bin/mpiexec.exe" -n 8 ./bin/hybrid_matrix.exe 1024

# 16-unit configs (full logical cores)
$env:OMP_NUM_THREADS = "16"; & "C:/MPI/Bin/mpiexec.exe" -n 1 ./bin/hybrid_matrix.exe 1024
$env:OMP_NUM_THREADS = "8";  & "C:/MPI/Bin/mpiexec.exe" -n 2 ./bin/hybrid_matrix.exe 1024
$env:OMP_NUM_THREADS = "4";  & "C:/MPI/Bin/mpiexec.exe" -n 4 ./bin/hybrid_matrix.exe 1024
$env:OMP_NUM_THREADS = "2";  & "C:/MPI/Bin/mpiexec.exe" -n 8 ./bin/hybrid_matrix.exe 1024
```

### What to point out during the demo
- "Hybrid covers both worlds — MPI ranks between (potentially across nodes) + OpenMP threads within."
- "Our best result is `2 procs × 8 threads`: 0.105 s, 20.48 GFLOPS — beats pure OpenMP by ~3 %."
- "Pattern: **more MPI ranks = more comm overhead**. Pure-MPI (`16p × 1t`) is the worst."
- "On a real cluster (multiple nodes), the hybrid model is the only option that scales properly — pure OpenMP can't cross nodes."

### Expected output pattern
```
Hybrid MPI+OpenMP GEMM | N=1024 | processes=2 | threads/proc=4
Total: 0.XXX s | Performance: XX.XX GFLOPS
Phase breakdown (per-phase max across ranks; % of phase-sum):
  Broadcast B : 0.XXX s ( X.X%)
  Scatter   A : 0.XXX s ( X.X%)
  Compute     : 0.XXX s ( X.X%)
  Gather    C : 0.XXX s ( X.X%)
  Comm total  : 0.XXX s ( X.X%)
```

### Show the source (highlight both layers)
```powershell
Get-Content src/hybrid_matrix.c | Select-String -Pattern "MPI_|#pragma omp" -Context 0,1
```

---

## Verifying correctness (anyone can run this)

After running variants, prove that every one produces output bit-identical to the serial baseline.

> **IMPORTANT:** all 5 output files in `outputs/` must be from runs at the **same N**. The RMSE script will refuse to compare matrices of different shapes.

### The clean way (recommended) — let `make verify` do everything
```powershell
mingw32-make verify N=1024 NP=8 THREADS=8
```
This runs all 5 variants at N=1024 then automatically prints the RMSE table at the end. **Best single demo command.**

### The manual way — if you've been running variants individually
```powershell
# First make sure all 5 variants have written outputs at the same N (say N=1024):
./bin/serial_gemm.exe 1024
$env:OMP_NUM_THREADS = "8"; ./bin/openmp_matrix.exe 1024
./bin/pthreads_matrix.exe 1024 8
& "C:/MPI/Bin/mpiexec.exe" -n 8 ./bin/mpi_matrix.exe 1024
$env:OMP_NUM_THREADS = "8"; & "C:/MPI/Bin/mpiexec.exe" -n 8 ./bin/hybrid_matrix.exe 1024

# Then run the RMSE check
python scripts/rmse_check.py
```

### Expected output (THE accuracy proof)
```
Reference: outputs\serial_output.txt
Implementation           RMSE      Max |err|  Status
------------------------------------------------------------
OpenMP         0.000000e+00   0.000000e+00  PASS (exact)
Pthreads       0.000000e+00   0.000000e+00  PASS (exact)
MPI            0.000000e+00   0.000000e+00  PASS (exact)
Hybrid         0.000000e+00   0.000000e+00  PASS (exact)
```

---

## Showing the analysis tables (optional final demo step)

After everything has been run at least once:

```powershell
python scripts/analyze_benchmarks.py
```

This prints three tables read from the saved CSVs in `results/`: strong scaling, problem-size sweep, hybrid configurations.

---

## Suggested division of work (3 group members, 5 variants)

| Member        | Variant(s) demonstrated |
|---------------|-------------------------|
| Adeesha       | Serial + Hybrid         |
| Akarshana     | OpenMP + Pthreads       |
| Akshayan      | MPI                     |

Recommended flow at the demo:
1. **Adeesha** opens with the **Serial** baseline → sets the reference.
2. **Akarshana** shows **OpenMP** then **Pthreads** → "OpenMP is just one pragma; here's the equivalent in raw threads".
3. **Akshayan** shows **MPI** → "Now we go distributed — note the phase breakdown".
4. **Adeesha** closes with **Hybrid** → "Putting it all together — best result is `2p × 8t`."
5. Anyone runs `mingw32-make verify N=1024 NP=8 THREADS=8` to finish with the **RMSE PASS table** — the accuracy proof.

Total demo runtime ≈ 5 minutes.

---

## If something goes wrong during demo

| Symptom | Quick fix |
|---|---|
| `'OMP_NUM_THREADS' is not recognized` | You used bash syntax. In PowerShell: `$env:OMP_NUM_THREADS = "8"` (separate line) |
| `Unexpected token '-n'` after `"C:/MPI/Bin/mpiexec.exe"` | Add `&` before the quoted path: `& "C:/MPI/..."` |
| `shape mismatch (1024, 1024) vs (512, 512)` | The 5 output files are from different N's. Run `mingw32-make verify N=1024 NP=8 THREADS=8` to regenerate all at the same N |
| `bin/serial_gemm.exe: not found` | You haven't built yet. Run `mingw32-make all` |
| Compilation errors | Run `mingw32-make clean` then `mingw32-make all` to do a fresh build |
