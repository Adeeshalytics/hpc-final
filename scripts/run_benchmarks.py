"""Benchmark driver for the HPC matrix-multiplication project.

Runs three sweeps and writes CSVs to <project-root>/results/ for the Analysis Report:
  - bench_strong_scaling.csv : threads/procs vary, N=1024
  - bench_problem_size.csv   : N varies, fixed unit counts
  - bench_hybrid.csv         : MPI x OpenMP configurations, N=1024

Each measurement is the minimum of REPEATS trials (best-of-N) to reduce noise.
All paths are resolved relative to the project root so the script works no matter
where it's invoked from.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPEATS = 3
N_FIXED = 1024
MPIEXEC = r"C:\MPI\Bin\mpiexec.exe"

ROOT       = Path(__file__).resolve().parent.parent
BIN_DIR    = ROOT / "bin"
RESULTS_DIR = ROOT / "results"

SERIAL   = str(BIN_DIR / "serial_gemm.exe")
OPENMP   = str(BIN_DIR / "openmp_matrix.exe")
PTHREADS = str(BIN_DIR / "pthreads_matrix.exe")
MPI_BIN  = str(BIN_DIR / "mpi_matrix.exe")
HYBRID   = str(BIN_DIR / "hybrid_matrix.exe")

TIME_RE   = re.compile(r"^Time:\s+([\d.]+)\s+s",  re.MULTILINE)
TOTAL_RE  = re.compile(r"^Total:\s+([\d.]+)\s+s", re.MULTILINE)
GFLOPS_RE = re.compile(r"Performance:\s+([\d.]+)\s+GFLOPS")
COMM_RE   = re.compile(r"Comm total\s*:\s+([\d.]+)")
COMP_RE   = re.compile(r"Compute\s+:\s+([\d.]+)")


def run_once(cmd, env=None):
    out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
    t = TIME_RE.search(out) or TOTAL_RE.search(out)
    g = GFLOPS_RE.search(out)
    c = COMM_RE.search(out)
    p = COMP_RE.search(out)
    return {
        "time":    float(t.group(1)) if t else None,
        "gflops":  float(g.group(1)) if g else None,
        "comm":    float(c.group(1)) if c else None,
        "compute": float(p.group(1)) if p else None,
    }


def best_of(cmd, env=None, n=REPEATS):
    runs = []
    for _ in range(n):
        try:
            runs.append(run_once(cmd, env=env))
        except subprocess.CalledProcessError as e:
            print(f"  FAIL: {' '.join(cmd)}\n    {e.output}", file=sys.stderr)
            return {"time": None, "gflops": None, "comm": None, "compute": None}
    return min(runs, key=lambda r: r["time"])


def env_threads(n):
    e = os.environ.copy()
    e["OMP_NUM_THREADS"] = str(n)
    return e


def csv_writer(fp, fields):
    w = csv.writer(fp)
    w.writerow(fields)
    return w


def fmt(x, p=4):
    return f"{x:.{p}f}" if isinstance(x, float) else ("-" if x is None else str(x))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)  # exes write there

    # ----- 1. Strong scaling at N=1024 ------------------------------------
    print(f"=== Strong scaling at N={N_FIXED} (best of {REPEATS}) ===")
    fields = ["variant", "units", "N", "time_s", "gflops", "comm_s", "compute_s"]
    with open(RESULTS_DIR / "bench_strong_scaling.csv", "w", newline="") as f:
        w = csv_writer(f, fields)

        r = best_of([SERIAL, str(N_FIXED)])
        w.writerow(["serial", 1, N_FIXED, fmt(r["time"], 6), fmt(r["gflops"], 3), "", ""])
        print(f"  serial          time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS")

        for n in (1, 2, 4, 8, 16):
            r = best_of([OPENMP, str(N_FIXED)], env=env_threads(n))
            w.writerow(["openmp", n, N_FIXED, fmt(r["time"], 6), fmt(r["gflops"], 3), "", ""])
            print(f"  openmp/{n:<2}       time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS")

        for n in (1, 2, 4, 8, 16):
            r = best_of([PTHREADS, str(N_FIXED), str(n)])
            w.writerow(["pthreads", n, N_FIXED, fmt(r["time"], 6), fmt(r["gflops"], 3), "", ""])
            print(f"  pthreads/{n:<2}     time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS")

        for n in (1, 2, 4, 8, 16):
            r = best_of([MPIEXEC, "-n", str(n), MPI_BIN, str(N_FIXED)])
            w.writerow(["mpi", n, N_FIXED, fmt(r["time"], 6), fmt(r["gflops"], 3),
                        fmt(r["comm"], 6), fmt(r["compute"], 6)])
            print(f"  mpi/{n:<2}          time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS"
                  f"  comm={fmt(r['comm'])}  comp={fmt(r['compute'])}")

    # ----- 2. Problem-size sweep -----------------------------------------
    print(f"\n=== Problem-size sweep (best of {REPEATS}) ===")
    with open(RESULTS_DIR / "bench_problem_size.csv", "w", newline="") as f:
        w = csv_writer(f, fields)
        for N in (256, 512, 1024, 2048):
            r = best_of([SERIAL, str(N)])
            w.writerow(["serial", 1, N, fmt(r["time"], 6), fmt(r["gflops"], 3), "", ""])
            print(f"  N={N:<4} serial     time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS")

            r = best_of([OPENMP, str(N)], env=env_threads(8))
            w.writerow(["openmp", 8, N, fmt(r["time"], 6), fmt(r["gflops"], 3), "", ""])
            print(f"  N={N:<4} openmp/8   time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS")

            r = best_of([MPIEXEC, "-n", "8", MPI_BIN, str(N)])
            w.writerow(["mpi", 8, N, fmt(r["time"], 6), fmt(r["gflops"], 3),
                        fmt(r["comm"], 6), fmt(r["compute"], 6)])
            print(f"  N={N:<4} mpi/8      time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS"
                  f"  comm={fmt(r['comm'])}  comp={fmt(r['compute'])}")

    # ----- 3. Hybrid configurations at N=1024 ----------------------------
    print(f"\n=== Hybrid configurations at N={N_FIXED} (best of {REPEATS}) ===")
    hfields = ["procs", "threads_per_proc", "total_units", "N", "time_s", "gflops", "comm_s", "compute_s"]
    configs = [
        (1, 8), (2, 4), (4, 2), (8, 1),                  # 8 logical units (matches physical cores)
        (1, 16), (2, 8), (4, 4), (8, 2), (16, 1),        # 16 logical units (full SMT)
    ]
    with open(RESULTS_DIR / "bench_hybrid.csv", "w", newline="") as f:
        w = csv_writer(f, hfields)
        for procs, tpp in configs:
            r = best_of(
                [MPIEXEC, "-n", str(procs), HYBRID, str(N_FIXED)],
                env=env_threads(tpp),
            )
            w.writerow([procs, tpp, procs * tpp, N_FIXED,
                        fmt(r["time"], 6), fmt(r["gflops"], 3),
                        fmt(r["comm"], 6), fmt(r["compute"], 6)])
            print(f"  {procs}p x {tpp}t (total {procs*tpp:>2})  "
                  f"time={fmt(r['time'])}s  {fmt(r['gflops'],2)} GFLOPS  "
                  f"comm={fmt(r['comm'])}  comp={fmt(r['compute'])}")

    print(f"\nWrote 3 CSVs to {RESULTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
