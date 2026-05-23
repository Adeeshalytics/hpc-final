"""Generate architecture / data-decomposition diagrams for the Analysis Report.

Outputs three PNGs into ./figures/:
  - 00_arch_decomposition.png : how A, B, C are split into row strips
  - 00_arch_mpi_dataflow.png  : MPI master-worker Bcast/Scatter/Compute/Gather
  - 00_arch_hybrid.png        : Hybrid MPI+OpenMP two-level hierarchy
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": False,
})

PROC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# =====================================================================
# Diagram 1: Domain decomposition (A row-strips, full B, C row-strips)
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 6)
ax.set_aspect("equal"); ax.axis("off")

def matrix(ax, x, y, w, h, label, strips=None, strip_colors=None, fill="#f0f0f0"):
    """Draw a matrix as a rectangle; optionally split into horizontal colored strips."""
    if strips is None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="black", linewidth=1.5))
    else:
        strip_h = h / strips
        for i in range(strips):
            ax.add_patch(Rectangle((x, y + i * strip_h), w, strip_h,
                                   facecolor=strip_colors[i], edgecolor="black",
                                   linewidth=1.0, alpha=0.7))
    ax.text(x + w / 2, y + h + 0.25, label, ha="center", va="bottom",
            fontsize=14, fontweight="bold")

# Three matrices: A, B, C
matrix(ax, 0.5, 1.0, 3.0, 4.0, "A  (split by rows)", strips=4, strip_colors=PROC_COLORS)
ax.text(4.0, 3.0, "×", ha="center", va="center", fontsize=24)

matrix(ax, 4.5, 1.0, 3.0, 4.0, "B  (broadcast to all)", fill="#cccccc")
ax.text(8.0, 3.0, "=", ha="center", va="center", fontsize=24)

matrix(ax, 8.5, 1.0, 3.0, 4.0, "C  (assembled from strips)", strips=4, strip_colors=PROC_COLORS)

# Process legend
for i, color in enumerate(PROC_COLORS):
    ax.add_patch(Rectangle((0.5 + i * 2.5, 0.1), 0.4, 0.3, facecolor=color, alpha=0.7,
                           edgecolor="black"))
    ax.text(1.0 + i * 2.5, 0.25, f"Process {i}", va="center", fontsize=11)

ax.set_title("Domain Decomposition: 1-D Row-Strip Partitioning of C = A × B",
             fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
fig.savefig(FIG / "00_arch_decomposition.png", bbox_inches="tight", dpi=150)
plt.close(fig)


# =====================================================================
# Diagram 2: MPI master-worker dataflow — 2x2 panel, one phase per panel
# =====================================================================
PHASE_COLORS = {
    "bcast":   "#2ca02c",
    "scatter": "#1f77b4",
    "compute": "#7f7f7f",
    "gather":  "#d62728",
}

def draw_ranks(ax, highlight=None):
    """Draw 4 rank boxes in a row; return list of (center_x, top_y, bottom_y, box_top)."""
    rank_w, rank_h = 1.6, 1.2
    centers = []
    for i in range(4):
        rx = 0.6 + i * 2.1
        ry = 0.7
        is_master = (i == 0)
        face = "#fff4e6" if is_master else "#f5f5f5"
        ax.add_patch(FancyBboxPatch((rx, ry), rank_w, rank_h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=face,
                                     edgecolor=PROC_COLORS[i], linewidth=1.5))
        label = f"Rank {i}"
        if is_master:
            label += "\n(Master)"
        ax.text(rx + rank_w / 2, ry + rank_h / 2, label,
                ha="center", va="center", fontsize=10.5)
        centers.append((rx + rank_w / 2, ry + rank_h, ry, ry + rank_h))
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return centers

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Panel ① — Bcast B
ax = axes[0, 0]
centers = draw_ranks(ax)
# arrows from rank 0 top to each other rank's top
for j in range(1, 4):
    a = FancyArrowPatch((centers[0][0], centers[0][1] + 0.15),
                        (centers[j][0], centers[j][1] + 0.05),
                        connectionstyle=f"arc3,rad=-0.35",
                        arrowstyle="-|>", mutation_scale=18,
                        color=PHASE_COLORS["bcast"], linewidth=2.2)
    ax.add_patch(a)
ax.text(4.5, 3.6, "① MPI_Bcast(B)   —   rank 0 sends the FULL matrix B to every rank",
        ha="center", fontsize=11, fontweight="bold",
        color=PHASE_COLORS["bcast"])

# Panel ② — Scatterv A
ax = axes[0, 1]
centers = draw_ranks(ax)
for j in range(4):
    a = FancyArrowPatch((centers[0][0], centers[0][1] + 0.05),
                        (centers[j][0], centers[j][1] + 0.05),
                        connectionstyle=("arc3,rad=0.0" if j == 0 else f"arc3,rad=0.35"),
                        arrowstyle="-|>", mutation_scale=18,
                        color=PHASE_COLORS["scatter"], linewidth=2.2,
                        linestyle=("--" if j == 0 else "-"))
    ax.add_patch(a)
ax.text(4.5, 3.6, "② MPI_Scatterv(A)   —   rank 0 sends each rank its OWN row-strip of A",
        ha="center", fontsize=11, fontweight="bold",
        color=PHASE_COLORS["scatter"])

# Panel ③ — Local compute
ax = axes[1, 0]
centers = draw_ranks(ax)
# small "gears" / compute label above each rank
for j in range(4):
    ax.text(centers[j][0], centers[j][1] + 0.3,
            "local_C\n= local_A·B",
            ha="center", fontsize=9, color=PHASE_COLORS["compute"],
            bbox=dict(facecolor="white", edgecolor=PHASE_COLORS["compute"],
                      boxstyle="round,pad=0.2"))
ax.text(4.5, 3.6, "③ Local Compute   —   every rank multiplies in parallel (no comm)",
        ha="center", fontsize=11, fontweight="bold",
        color=PHASE_COLORS["compute"])

# Panel ④ — Gatherv C
ax = axes[1, 1]
centers = draw_ranks(ax)
for j in range(4):
    a = FancyArrowPatch((centers[j][0], centers[j][1] + 0.05),
                        (centers[0][0], centers[0][1] + 0.05),
                        connectionstyle=("arc3,rad=0.0" if j == 0 else f"arc3,rad=-0.35"),
                        arrowstyle="-|>", mutation_scale=18,
                        color=PHASE_COLORS["gather"], linewidth=2.2,
                        linestyle=("--" if j == 0 else "-"))
    ax.add_patch(a)
ax.text(4.5, 3.6, "④ MPI_Gatherv(C)   —   every rank sends its strip of C back to rank 0",
        ha="center", fontsize=11, fontweight="bold",
        color=PHASE_COLORS["gather"])

fig.suptitle("MPI Master-Worker Dataflow — 4 Phases (shown with 4 ranks)",
             fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG / "00_arch_mpi_dataflow.png", bbox_inches="tight", dpi=150)
plt.close(fig)


# =====================================================================
# Diagram 3: Hybrid two-level hierarchy
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 7)
ax.set_aspect("equal"); ax.axis("off")

# Two MPI processes, each containing 4 OpenMP threads
for p in range(2):
    px = 1.0 + p * 6.5
    # Outer MPI process box
    ax.add_patch(FancyBboxPatch((px, 1.0), 5.5, 5.0, boxstyle="round,pad=0.1",
                                 facecolor="#e8f0fe", edgecolor="#1f77b4", linewidth=2.2))
    ax.text(px + 2.75, 5.7, f"MPI Process {p}", ha="center", fontsize=13, fontweight="bold",
            color="#1f77b4")
    ax.text(px + 2.75, 5.3, "(owns row-strip of A, full B, row-strip of C)",
            ha="center", fontsize=9.5, style="italic", color="#444")

    # Inner OpenMP threads (4 per process)
    for t in range(4):
        tx = px + 0.4 + t * 1.25
        ax.add_patch(FancyBboxPatch((tx, 1.5), 1.0, 2.8, boxstyle="round,pad=0.04",
                                     facecolor="#fff4e6", edgecolor="#ff7f0e", linewidth=1.2))
        ax.text(tx + 0.5, 3.6, f"OMP\nThread {t}", ha="center", va="center",
                fontsize=10, color="#ff7f0e", fontweight="bold")
        ax.text(tx + 0.5, 2.4, "rows\n[a:b)", ha="center", va="center", fontsize=8.5, color="#555")

# Network arrow between processes
arrow_mpi = FancyArrowPatch((6.5, 3.5), (7.5, 3.5),
                             arrowstyle="<|-|>", mutation_scale=22,
                             color="#2ca02c", linewidth=2.5)
ax.add_patch(arrow_mpi)
ax.text(7.0, 3.95, "MPI", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")
ax.text(7.0, 3.05, "(network /\nshared mem)", ha="center", fontsize=8.5, color="#2ca02c")

ax.text(7.0, 0.5, "Hierarchy:  MPI between processes  +  OpenMP threads within each process",
        ha="center", fontsize=11, fontweight="bold",
        bbox=dict(facecolor="#fff7e6", edgecolor="#ff7f0e", boxstyle="round,pad=0.4"))

ax.set_title("Hybrid MPI + OpenMP — Two-Level Parallel Hierarchy",
             fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
fig.savefig(FIG / "00_arch_hybrid.png", bbox_inches="tight", dpi=150)
plt.close(fig)


print("Wrote architecture diagrams:")
for p in sorted(FIG.glob("00_arch_*.png")):
    print(f"  {p.name}")
