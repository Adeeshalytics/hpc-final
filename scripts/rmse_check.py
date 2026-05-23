"""Accuracy verification: compare each parallel implementation against the serial baseline.

Reads output text files from <project-root>/outputs/. Works no matter where it's invoked from.
"""

import sys
from pathlib import Path
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"


def compare(reference_file: Path, test_file: Path):
    if not test_file.exists():
        return None, None, "missing"
    R = np.loadtxt(reference_file)
    T = np.loadtxt(test_file)
    if R.shape != T.shape:
        return None, None, f"shape mismatch {R.shape} vs {T.shape}"
    diff = R - T
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    return rmse, max_abs, "ok"


def main() -> int:
    ref = OUT_DIR / "serial_output.txt"
    if not ref.exists():
        print(f"ERROR: reference file '{ref}' not found. Run the serial program first.")
        return 1

    targets = [
        ("OpenMP",   OUT_DIR / "openmp_output.txt"),
        ("Pthreads", OUT_DIR / "pthreads_output.txt"),
        ("MPI",      OUT_DIR / "mpi_output.txt"),
        ("Hybrid",   OUT_DIR / "hybrid_output.txt"),
    ]

    print(f"Reference: {ref.relative_to(ROOT)}")
    print(f"{'Implementation':<12} {'RMSE':>14} {'Max |err|':>14}  Status")
    print("-" * 60)

    failures = 0
    for name, fpath in targets:
        rmse, max_abs, status = compare(ref, fpath)
        if status == "ok":
            if rmse == 0.0:
                tag = "PASS (exact)"
            elif max_abs < 1e-6:
                tag = "PASS (FP drift)"
            else:
                tag = "FAIL"
                failures += 1
            print(f"{name:<12} {rmse:>14.6e} {max_abs:>14.6e}  {tag}")
        else:
            print(f"{name:<12} {'-':>14} {'-':>14}  {status}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
