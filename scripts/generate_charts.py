"""Generate all charts for the Analysis Report from the bench_*.csv files.

Outputs PNGs into the ./figures/ directory. Run after `run_benchmarks.py`
and `analyze_benchmarks.py` have produced the CSVs.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi":   120,
    "savefig.dpi":  150,
    "font.size":     11,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG     = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def read(path):
    with open(RESULTS / path, newline="") as f:
        return list(csv.DictReader(f))


def num(s):
    return float(s) if s not in ("", None) else None


def by_variant(rows, variant):
    sel = [r for r in rows if r["variant"] == variant]
    sel.sort(key=lambda r: int(r["units"]))
    return sel


# ------------------------------------------------------------------------
# Load all CSVs
# ------------------------------------------------------------------------
ss   = read("bench_strong_scaling.csv")
ps   = read("bench_problem_size.csv")
hy   = read("bench_hybrid.csv")
spd  = read("bench_speedup_efficiency.csv")

serial_t = next(num(r["time_s"]) for r in ss if r["variant"] == "serial")


# ------------------------------------------------------------------------
# Chart 1: Speedup vs cores (strong scaling)
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
cores = [1, 2, 4, 8, 16]
ax.plot(cores, cores, "--", color="grey", label="Ideal (linear)", alpha=0.7)

for variant, color, marker in [("openmp", "#1f77b4", "o"),
                                ("pthreads", "#ff7f0e", "s"),
                                ("mpi", "#2ca02c", "^")]:
    rows = by_variant(spd, variant)
    units = [int(r["units"]) for r in rows]
    sp    = [num(r["speedup"]) for r in rows]
    ax.plot(units, sp, marker=marker, color=color, label=variant.capitalize())

ax.set_xscale("log", base=2)
ax.set_xticks(cores)
ax.set_xticklabels(cores)
ax.set_xlabel("Number of cores / threads / processes")
ax.set_ylabel("Speedup (vs serial)")
ax.set_title("Strong Scaling at N=1024 — Speedup")
ax.legend(loc="upper left")
ax.set_xlim(0.9, 18)
fig.tight_layout()
fig.savefig(FIG / "01_speedup_strong_scaling.png", bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------------
# Chart 2: Efficiency vs cores
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.axhline(100, color="grey", ls="--", alpha=0.7, label="Ideal (100%)")

for variant, color, marker in [("openmp", "#1f77b4", "o"),
                                ("pthreads", "#ff7f0e", "s"),
                                ("mpi", "#2ca02c", "^")]:
    rows = by_variant(spd, variant)
    units = [int(r["units"]) for r in rows]
    ef    = [num(r["efficiency_pct"]) for r in rows]
    ax.plot(units, ef, marker=marker, color=color, label=variant.capitalize())

ax.set_xscale("log", base=2)
ax.set_xticks(cores); ax.set_xticklabels(cores)
ax.set_xlabel("Number of cores / threads / processes")
ax.set_ylabel("Parallel Efficiency (%)")
ax.set_title("Strong Scaling at N=1024 — Parallel Efficiency")
ax.legend(loc="upper right")
ax.set_ylim(0, 110)
fig.tight_layout()
fig.savefig(FIG / "02_efficiency_strong_scaling.png", bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------------
# Chart 3: Execution time vs cores (log-log)
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.axhline(serial_t, color="black", ls=":", alpha=0.6, label=f"Serial ({serial_t:.3f} s)")

for variant, color, marker in [("openmp", "#1f77b4", "o"),
                                ("pthreads", "#ff7f0e", "s"),
                                ("mpi", "#2ca02c", "^")]:
    rows = by_variant(ss, variant)
    units = [int(r["units"]) for r in rows]
    t     = [num(r["time_s"]) for r in rows]
    ax.plot(units, t, marker=marker, color=color, label=variant.capitalize())

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(cores); ax.set_xticklabels(cores)
ax.set_xlabel("Number of cores / threads / processes")
ax.set_ylabel("Execution Time (s, log scale)")
ax.set_title("Strong Scaling at N=1024 — Execution Time")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "03_time_strong_scaling.png", bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------------
# Chart 4: GFLOPS vs problem size (the "memory wall")
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

for variant, color, marker, label in [
    ("serial", "#d62728", "o", "Serial (1 core)"),
    ("openmp", "#1f77b4", "s", "OpenMP (8 threads)"),
    ("mpi",    "#2ca02c", "^", "MPI (8 processes)"),
]:
    rows = [r for r in ps if r["variant"] == variant]
    rows.sort(key=lambda r: int(r["N"]))
    Ns    = [int(r["N"]) for r in rows]
    gfs   = [num(r["gflops"]) for r in rows]
    ax.plot(Ns, gfs, marker=marker, color=color, label=label)

ax.set_xscale("log", base=2)
ax.set_xticks([256, 512, 1024, 2048])
ax.set_xticklabels(["256", "512", "1024", "2048"])
ax.set_xlabel("Matrix dimension N")
ax.set_ylabel("Performance (GFLOPS)")
ax.set_title("Performance vs Problem Size — the Memory Wall")
ax.legend(loc="lower left")
# annotation for cache size
l3_threshold = "L3 cache (16 MB) ≈ N ~ 729"
ax.axvline(729, color="grey", ls="--", alpha=0.5)
ax.text(740, 22, "L3 working\nset (~16 MB)", fontsize=9, color="grey")
fig.tight_layout()
fig.savefig(FIG / "04_gflops_vs_N.png", bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------------
# Chart 5: Hybrid configuration comparison
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))

hy_sorted = sorted(hy, key=lambda r: (int(r["total_units"]), int(r["procs"])))
labels   = [f"{r['procs']}p×{r['threads_per_proc']}t" for r in hy_sorted]
times    = [num(r["time_s"]) for r in hy_sorted]
total_u  = [int(r["total_units"]) for r in hy_sorted]
colors   = ["#1f77b4" if u == 8 else "#ff7f0e" for u in total_u]

bars = ax.bar(range(len(labels)), times, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=0)
ax.set_ylabel("Execution Time (s)")
ax.set_title("Hybrid MPI×OpenMP Configurations at N=1024")

# annotate values on each bar
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{t:.3f}", ha="center", va="bottom", fontsize=9)

# best config marker
best_idx = times.index(min(times))
bars[best_idx].set_edgecolor("red")
bars[best_idx].set_linewidth(2.5)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#1f77b4", label="8 total units"),
    Patch(color="#ff7f0e", label="16 total units"),
    Patch(facecolor="white", edgecolor="red", linewidth=2.5, label="Best"),
], loc="upper left")

ax.set_ylim(0, max(times) * 1.15)
fig.tight_layout()
fig.savefig(FIG / "05_hybrid_configs.png", bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------------
# Chart 6: MPI phase breakdown (stacked: compute vs comm)
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

mpi_rows = by_variant(ss, "mpi")
procs    = [int(r["units"]) for r in mpi_rows]
comp_t   = [num(r["compute_s"]) for r in mpi_rows]
comm_t   = [num(r["comm_s"])    for r in mpi_rows]

x = np.arange(len(procs))
width = 0.6
bars_comp = ax.bar(x, comp_t, width, label="Compute", color="#2ca02c")
bars_comm = ax.bar(x, comm_t, width, bottom=comp_t, label="Communication (bcast+scatter+gather)",
                   color="#d62728")

ax.set_xticks(x)
ax.set_xticklabels([str(p) for p in procs])
ax.set_xlabel("MPI processes")
ax.set_ylabel("Time (s) — sum of phase maxes across ranks")
ax.set_title("MPI Phase Breakdown at N=1024 — Communication vs Compute")
ax.legend(loc="upper left")

# label compute fraction on each bar
for i, (c, m) in enumerate(zip(comp_t, comm_t)):
    total = c + m
    ax.text(i, total + 0.005, f"{100*c/total:.0f}% / {100*m/total:.0f}%",
            ha="center", fontsize=9, color="black")

fig.tight_layout()
fig.savefig(FIG / "06_mpi_phase_breakdown.png", bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------------
# Chart 7: All variants head-to-head at the best 8-core configuration
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

# pull representative best-at-8 result for each
best = {
    "Serial":     next(num(r["time_s"]) for r in ss if r["variant"] == "serial"),
    "OpenMP/8":   next(num(r["time_s"]) for r in ss if r["variant"] == "openmp" and int(r["units"]) == 8),
    "Pthreads/8": next(num(r["time_s"]) for r in ss if r["variant"] == "pthreads" and int(r["units"]) == 8),
    "MPI/8":      next(num(r["time_s"]) for r in ss if r["variant"] == "mpi" and int(r["units"]) == 8),
    "Hybrid 2p×8t": next(num(r["time_s"]) for r in hy
                          if int(r["procs"]) == 2 and int(r["threads_per_proc"]) == 8),
}

names = list(best.keys())
vals  = list(best.values())
colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
bars = ax.bar(names, vals, color=colors)
for bar, v in zip(bars, vals):
    sp = serial_t / v
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}s\n({sp:.2f}×)", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Execution Time (s)")
ax.set_title("Head-to-Head Comparison at N=1024 (best 8/16-unit config)")
ax.set_ylim(0, max(vals) * 1.20)
fig.tight_layout()
fig.savefig(FIG / "07_head_to_head.png", bbox_inches="tight")
plt.close(fig)


print(f"Wrote 7 charts to {FIG}/")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name}")
