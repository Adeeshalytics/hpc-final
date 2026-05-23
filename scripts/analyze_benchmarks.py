"""Derive speedup / efficiency tables and a markdown summary from the bench CSVs."""

import csv
from pathlib import Path


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(s):
    return float(s) if s not in ("", None) else None


def main():
    root = Path(__file__).resolve().parent.parent / "results"
    ss   = read(root / "bench_strong_scaling.csv")
    ps   = read(root / "bench_problem_size.csv")
    hy   = read(root / "bench_hybrid.csv")

    # Serial baseline at N=1024 (for speedup/efficiency)
    serial_t = next(float(r["time_s"]) for r in ss if r["variant"] == "serial")

    # -------- Speedup & efficiency at N=1024 --------
    rows = []
    for r in ss:
        t = num(r["time_s"])
        if not t:
            continue
        units = int(r["units"])
        speedup    = serial_t / t
        efficiency = 100 * speedup / units
        rows.append({
            "variant":    r["variant"],
            "units":      units,
            "time_s":     t,
            "gflops":     num(r["gflops"]),
            "speedup":    speedup,
            "efficiency": efficiency,
        })

    with open(root / "bench_speedup_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "units", "time_s", "gflops", "speedup", "efficiency_pct"])
        for r in rows:
            w.writerow([r["variant"], r["units"], f"{r['time_s']:.6f}", f"{r['gflops']:.3f}",
                        f"{r['speedup']:.3f}", f"{r['efficiency']:.2f}"])

    # Hybrid speedup
    hyb_rows = []
    for r in hy:
        t = num(r["time_s"])
        if not t:
            continue
        units = int(r["total_units"])
        speedup    = serial_t / t
        efficiency = 100 * speedup / units
        hyb_rows.append({
            "procs":      int(r["procs"]),
            "threads":    int(r["threads_per_proc"]),
            "units":      units,
            "time_s":     t,
            "gflops":     num(r["gflops"]),
            "speedup":    speedup,
            "efficiency": efficiency,
        })

    with open(root / "bench_hybrid_speedup.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["procs", "threads_per_proc", "total_units", "time_s", "gflops", "speedup", "efficiency_pct"])
        for r in hyb_rows:
            w.writerow([r["procs"], r["threads"], r["units"], f"{r['time_s']:.6f}",
                        f"{r['gflops']:.3f}", f"{r['speedup']:.3f}", f"{r['efficiency']:.2f}"])

    # -------- Print summary tables --------
    def line():
        print("-" * 78)

    print("\n## Speedup & Efficiency at N=1024 (serial baseline = {:.4f}s)".format(serial_t))
    print(f"{'Variant':<10} {'Units':>6} {'Time(s)':>10} {'GFLOPS':>8} {'Speedup':>9} {'Eff. (%)':>10}")
    line()
    for r in rows:
        print(f"{r['variant']:<10} {r['units']:>6} {r['time_s']:>10.4f} "
              f"{r['gflops']:>8.2f} {r['speedup']:>9.2f} {r['efficiency']:>10.1f}")

    print("\n## Problem-size sweep (GFLOPS shown to highlight cache regime)")
    print(f"{'Variant':<10} {'Units':>6} {'N':>6} {'Time(s)':>10} {'GFLOPS':>8}")
    line()
    for r in ps:
        t = num(r["time_s"])
        g = num(r["gflops"])
        if t is None: continue
        print(f"{r['variant']:<10} {r['units']:>6} {r['N']:>6} {t:>10.4f} {g:>8.2f}")

    print("\n## Hybrid configurations at N=1024")
    print(f"{'Procs':>5} {'x Thr':>5} {'Total':>6} {'Time(s)':>10} {'GFLOPS':>8} "
          f"{'Speedup':>9} {'Eff.(%)':>9} {'Comm(s)':>9} {'Comp(s)':>9}")
    line()
    for r, hr in zip(hy, hyb_rows):
        comm = num(r["comm_s"]); comp = num(r["compute_s"])
        print(f"{hr['procs']:>5} {hr['threads']:>5} {hr['units']:>6} {hr['time_s']:>10.4f} "
              f"{hr['gflops']:>8.2f} {hr['speedup']:>9.2f} {hr['efficiency']:>9.1f} "
              f"{(comm or 0):>9.4f} {(comp or 0):>9.4f}")

    print("\nWrote: bench_speedup_efficiency.csv, bench_hybrid_speedup.csv")


if __name__ == "__main__":
    main()
