"""Generates the report figures from results/summary.csv and results/runs/.

  python scripts/plot_results.py     ->  results/plots/*.png   (or $RESULTS_DIR/plots/*.png)
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results")).resolve()
PLOTS = RESULTS / "plots"

# Validated categorical palette (fixed order, never cycled) and chart chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

DATASET_ORDER = ["web-Google", "web-BerkStan", "soc-LiveJournal1", "rmat-22"]
MODE_ORDER = ["allgather", "alltoallv", "overlap"]
MAIN_MODE, MAIN_PART = "alltoallv", "edge"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})
MARKER = dict(marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)


def color_for(key, order):
    return SERIES[order.index(key) % len(SERIES)] if key in order else MUTED


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PLOTS / name)


def datasets_in(df):
    return [d for d in DATASET_ORDER if d in set(df.dataset)] + sorted(set(df.dataset) - set(DATASET_ORDER))


def main_config(df):
    return df[(df.program == "mpi") & (df["mode"] == MAIN_MODE) & (df.partition == MAIN_PART)]


def ideal_line(ax, procs):
    ax.plot(procs, procs, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("ideal", (procs[-1], procs[-1]), xytext=(4, 0), textcoords="offset points",
                color=MUTED, fontsize=9, va="center")


def physical_core_marker(ax, cores=4):
    ax.axvline(cores + 0.5, color=AXIS, linewidth=1, zorder=0)
    ax.text(cores + 0.6, ax.get_ylim()[1], "hyper-threads →", color=MUTED, fontsize=8, va="top")


def plot_speedup_efficiency(df):
    strong = main_config(df[df.kind == "strong"])
    if strong.empty:
        return
    procs = sorted(strong.procs.unique())
    order = datasets_in(strong)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ideal_line(ax1, procs)
    for ds in order:
        s = strong[strong.dataset == ds].sort_values("procs")
        c = color_for(ds, order)
        ax1.plot(s.procs, s.speedup, color=c, label=ds, **MARKER)
        ax2.plot(s.procs, s.efficiency * 100, color=c, label=ds, **MARKER)
    ax1.set(title="Speedup vs sequential (solve phase)", xlabel="MPI processes", ylabel="speedup  T_seq / T_p",
            xticks=procs)
    ax2.set(title="Parallel efficiency", xlabel="MPI processes", ylabel="efficiency (%)", xticks=procs, ylim=(0, 110))
    for ax in (ax1, ax2):
        physical_core_marker(ax)
    ax1.legend(loc="upper left")
    fig.suptitle(f"Strong scaling - mode {MAIN_MODE}, {MAIN_PART}-balanced partition", x=0.01, ha="left",
                 color=INK2, fontsize=9)
    save(fig, "speedup_efficiency.png")


def plot_exec_time(df):
    strong = df[df.kind == "strong"]
    order = datasets_in(strong)
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], order):
        mpi = main_config(strong[strong.dataset == ds]).sort_values("procs")
        seq = strong[(strong.dataset == ds) & (strong.program == "seq")]
        ax.bar(mpi.procs.astype(str), mpi.t_solve, width=0.55, color=SERIES[0], zorder=2)
        if not seq.empty:
            t = seq.t_solve.iloc[0]
            ax.axhline(t, color=SERIES[1], linewidth=2, zorder=3)
            ax.annotate(f"sequential {t:.2f}s", (0, t), xytext=(0, 4), textcoords="offset points",
                        color=INK2, fontsize=8)
        for x, y in zip(mpi.procs.astype(str), mpi.t_solve):
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 2), textcoords="offset points", ha="center",
                        color=INK2, fontsize=7.5)
        ax.set(title=ds, xlabel="processes", ylabel="solve time (s)" if ax is axes[0][0] else "")
        ax.grid(axis="x", visible=False)
    save(fig, "execution_time.png")


def plot_modes(df):
    strong = df[(df.kind == "strong") & (df.program == "mpi") & (df.partition == MAIN_PART)]
    order = datasets_in(strong)
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.4), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], order):
        s = strong[strong.dataset == ds]
        procs = sorted(s.procs.unique())
        ideal_line(ax, procs)
        for mode in MODE_ORDER:
            m = s[s["mode"] == mode].sort_values("procs")
            ax.plot(m.procs, m.speedup, color=color_for(mode, MODE_ORDER), label=mode, **MARKER)
        ax.set(title=ds, xlabel="processes", xticks=procs)
    axes[0][0].set_ylabel("speedup")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Communication strategy comparison (edge-balanced partition)", x=0.01, ha="left", color=INK2,
                 fontsize=9)
    save(fig, "comm_modes.png")


def plot_breakdown(df):
    """Average per-rank time split into compute / exchange / reduce, for each mode at max P."""
    strong = df[(df.kind == "strong") & (df.program == "mpi") & (df.partition == MAIN_PART)]
    order = datasets_in(strong)
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(3.4 * len(order), 3.6), squeeze=False)
    parts = [("t_compute_avg", "compute"), ("t_exchange_avg", "exchange (data)"), ("t_reduce_avg", "allreduce + wait")]
    abbr = {"allgather": "AG", "alltoallv": "A2A", "overlap": "OV"}
    for ax, ds in zip(axes[0], order):
        s = strong[strong.dataset == ds]
        rows = s[s.procs.isin([2, 4, 8])].sort_values(["procs", "mode"],
                                                      key=lambda c: c.map(MODE_ORDER.index) if c.name == "mode" else c)
        procs = sorted(rows.procs.unique())
        # Bars grouped by P with a gap between groups.
        xs = [procs.index(p) * (len(MODE_ORDER) + 1) + MODE_ORDER.index(m) for p, m in zip(rows.procs, rows["mode"])]
        bottom = [0.0] * len(rows)
        for i, (col, name) in enumerate(parts):
            vals = rows[col].tolist()
            ax.bar(xs, vals, bottom=bottom, width=0.75, color=SERIES[i], label=name,
                   edgecolor=SURFACE, linewidth=1, zorder=2)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(xs, [abbr[m] for m in rows["mode"]], fontsize=7.5)
        for k, p in enumerate(procs):
            ax.annotate(f"P = {p}", (k * (len(MODE_ORDER) + 1) + 1, 0), xytext=(0, -26), textcoords="offset points",
                        xycoords=("data", "axes fraction"), ha="center", color=INK2, fontsize=8.5)
        ax.set(title=ds, ylabel="time per rank (s)" if ax is axes[0][0] else "")
        ax.grid(axis="x", visible=False)
    axes[0][0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Where the time goes (AG = allgather, A2A = alltoallv, OV = overlap; edge-balanced partition)",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    save(fig, "time_breakdown.png")


def plot_comm_volume(df):
    strong = df[(df.kind == "strong") & (df.program == "mpi") & (df.partition == MAIN_PART) & (df.procs > 1)]
    order = datasets_in(strong)
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.3), squeeze=False)
    for ax, ds in zip(axes[0], order):
        s = strong[strong.dataset == ds]
        for mode in ["allgather", "alltoallv"]:
            m = s[s["mode"] == mode].sort_values("procs")
            ax.plot(m.procs, m.bytes_per_iter / 1e6, color=color_for(mode, MODE_ORDER), label=mode, **MARKER)
        ax.set(title=ds, xlabel="processes", xticks=sorted(s.procs.unique()))
    axes[0][0].set_ylabel("MB moved per iteration")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    save(fig, "comm_volume.png")


def plot_partition_balance(df):
    """Imbalance factor (max/avg compute) vs P, vertex vs edge partitioning."""
    strong = df[(df.kind == "strong") & (df.program == "mpi") & (df["mode"] == MAIN_MODE) & (df.procs > 1)]
    order = datasets_in(strong)
    if not order:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 3.3), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], order):
        s = strong[strong.dataset == ds]
        for i, part in enumerate(["vertex", "edge"]):
            m = s[s.partition == part].sort_values("procs")
            ax.plot(m.procs, m.imbalance, color=SERIES[i], label=f"{part} partition", **MARKER)
        ax.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set(title=ds, xlabel="processes", xticks=sorted(s.procs.unique()))
    axes[0][0].set_ylabel("compute imbalance (max / avg)")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    save(fig, "load_imbalance.png")


def plot_per_rank_work():
    """Per-rank in-edge counts at P = 8 for both partitioners (one panel per dataset)."""
    panels = []
    for ds in DATASET_ORDER:
        files = {p: RESULTS / "runs" / ds / f"mpi_p8_{MAIN_MODE}_{p}_r0.json" for p in ["vertex", "edge"]}
        if all(f.exists() for f in files.values()):
            panels.append((ds, {p: json.loads(f.read_text())["per_rank"] for p, f in files.items()}))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.3), squeeze=False)
    for ax, (ds, runs) in zip(axes[0], panels):
        P = len(runs["vertex"]["edges"])
        w = 0.38
        for i, part in enumerate(["vertex", "edge"]):
            xs = [r + (i - 0.5) * (w + 0.04) for r in range(P)]
            ax.bar(xs, [e / 1e6 for e in runs[part]["edges"]], width=w, color=SERIES[i], label=f"{part} partition",
                   zorder=2)
        ax.set(title=ds, xlabel="rank", xticks=range(P))
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("in-edges owned (millions)")
    axes[0][0].legend(loc="upper right", fontsize=8.5)
    save(fig, "per_rank_edges.png")


def plot_convergence():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plotted = False
    for i, ds in enumerate(DATASET_ORDER):
        f = RESULTS / "runs" / ds / "seq_r0.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())["residuals"]
        ax.semilogy(range(1, len(res) + 1), res, color=SERIES[i], label=f"{ds} ({len(res)} it.)")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(1e-9, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("tolerance 1e-9", (1, 1e-9), xytext=(2, 4), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.set(title="Convergence of the power iteration", xlabel="iteration", ylabel="L1 residual  ||r' - r||₁")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "convergence.png")


def plot_weak(df):
    weak = df[df.kind == "weak"].sort_values("procs")
    if weak.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = [f"{p}\n{d}" for p, d in zip(weak.procs, weak.dataset)]
    ax1.bar(labels, weak.t_solve, width=0.5, color=SERIES[0], zorder=2)
    ax1.set(title="Weak scaling - solve time (30 iterations)", ylabel="seconds")
    ax1.grid(axis="x", visible=False)
    ax2.plot(weak.procs, weak.efficiency * 100, color=SERIES[0], **MARKER)
    ax2.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax2.set(title="Weak-scaling efficiency  T(1) / T(P)", xlabel="processes", ylabel="%",
            xticks=list(weak.procs), ylim=(0, 110))
    save(fig, "weak_scaling.png")


def plot_platform(df):
    """Speedup of the hardware probe kernels (scripts/probe_platform.sh) next to PageRank."""
    f = RESULTS / "platform.csv"
    if not f.exists():
        return
    probe = pd.read_csv(f).groupby("procs", as_index=False).median()
    base = probe[probe.procs == 1].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    procs = list(probe.procs)
    ideal_line(ax, procs)
    kernels = [("t_compute", "compute-bound kernel"), ("t_gather", "random-gather kernel"),
               ("t_stream", "memory-streaming kernel")]
    for i, (col, name) in enumerate(kernels):
        ax.plot(probe.procs, base[col] / probe[col], color=SERIES[i], label=name, **MARKER)
    best = df[(df.kind == "strong") & (df.program == "mpi")].groupby(["dataset", "procs"], as_index=False).speedup.max()
    for ds, style in [("rmat-22", (0, (1, 2))), ("soc-LiveJournal1", (0, (5, 2)))]:
        b = best[best.dataset == ds].sort_values("procs")
        ax.plot(b.procs, b.speedup, color=INK2, linewidth=1.5, linestyle=style, label=f"PageRank, {ds} (best)")
    ax.set(title="The machine's own ceiling: speedup of simple kernels", xlabel="MPI processes",
           ylabel="speedup over 1 process", xticks=procs)
    physical_core_marker(ax)
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "platform_ceiling.png")


def main():
    df = pd.read_csv(RESULTS / "summary.csv")
    plot_platform(df)
    plot_speedup_efficiency(df)
    plot_exec_time(df)
    plot_modes(df)
    plot_breakdown(df)
    plot_comm_volume(df)
    plot_partition_balance(df)
    plot_per_rank_work()
    plot_convergence()
    plot_weak(df)


if __name__ == "__main__":
    main()
