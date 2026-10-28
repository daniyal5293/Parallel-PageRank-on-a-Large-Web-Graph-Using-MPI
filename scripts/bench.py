"""Benchmark driver: runs the sequential and MPI PageRank over datasets,
process counts, communication modes and partitioners, then aggregates.

  python scripts/bench.py                 # full sweep (resumable: finished runs are skipped)
  python scripts/bench.py --quick         # small sweep on web-Google only
  python scripts/bench.py --aggregate     # only rebuild the CSV files from results/runs/

Outputs
  (all under results/, or under $RESULTS_DIR when that is set)
  results/runs/<dataset>/<config>_r<k>.json   raw output of every run
  results/bench.csv                           one row per run
  results/summary.csv                         median per configuration + speedup/efficiency
  results/system.json                         hardware / software description
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results")).resolve()
RUNS = RESULTS / "runs"
BUILD = ROOT / "build"
DATA_DIR = Path(os.environ.get("DATA_DIR", Path.home() / "pdc-data"))
PHYSICAL_CORES = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1

STRONG_DATASETS = ["web-Google", "web-BerkStan", "soc-LiveJournal1", "rmat-22"]
WEAK_SERIES = {1: "rmat-20", 2: "rmat-21", 4: "rmat-22", 8: "rmat-23"}  # ~constant edges per process
MODES = ["allgather", "alltoallv", "overlap"]
PARTITIONS = ["vertex", "edge"]


def mpirun_cmd(p):
    """One rank per physical core while they last; beyond that, hyper-threads."""
    if p <= PHYSICAL_CORES:
        return ["mpirun", "-np", str(p), "--map-by", "core", "--bind-to", "core"]
    # Round-robin over cores: ranks 0-3 get hw-thread 0 of each core, ranks 4+ the siblings.
    return ["mpirun", "-np", str(p), "--use-hwthread-cpus", "--map-by", "core", "--bind-to", "hwthread"]


def run_one(cmd, json_path, timeout):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd + ["--json", str(json_path)], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not json_path.exists():
        print(f"    FAILED ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
        return None
    data = json.loads(json_path.read_text())
    print(f"    {json_path.stem:<36} solve {data['t_solve']:8.3f}s  iters {data['iterations']:3d}  "
          f"(wall {time.time() - t0:.1f}s)")
    return data


def sweep(args):
    datasets = args.datasets.split(",") if args.datasets else (["web-Google"] if args.quick else STRONG_DATASETS)
    procs = [int(x) for x in args.procs.split(",")] if args.procs else ([1, 2, 4] if args.quick else [1, 2, 3, 4, 6, 8])
    modes = args.modes.split(",") if args.modes else MODES
    partitions = args.partitions.split(",") if args.partitions else PARTITIONS
    repeats = args.repeats or (1 if args.quick else 3)
    extra = ["--topk", "20"] + (["--iters", str(args.iters)] if args.iters else [])

    for ds in datasets:
        graph = DATA_DIR / f"{ds}.bin"
        if not graph.exists():
            print(f"skip {ds}: {graph} not found (run scripts/prepare_datasets.sh)")
            continue
        print(f"== strong scaling: {ds}")
        for r in range(repeats):
            out = RUNS / ds / f"seq_r{r}.json"
            if not out.exists() or args.force:
                run_one([str(BUILD / "pagerank_seq"), "--graph", str(graph)] + extra, out, args.timeout)
        for p in procs:
            for mode in modes:
                for part in partitions:
                    for r in range(repeats):
                        out = RUNS / ds / f"mpi_p{p}_{mode}_{part}_r{r}.json"
                        if out.exists() and not args.force:
                            continue
                        cmd = mpirun_cmd(p) + [str(BUILD / "pagerank_mpi"), "--graph", str(graph),
                                               "--mode", mode, "--partition", part] + extra
                        run_one(cmd, out, args.timeout)

    if args.weak and not args.quick:
        print("== weak scaling (R-MAT, ~constant edges per process)")
        for p, ds in WEAK_SERIES.items():
            graph = DATA_DIR / f"{ds}.bin"
            if not graph.exists():
                print(f"skip weak p={p}: {graph} not found")
                continue
            for r in range(repeats):
                out = RUNS / "weak" / f"mpi_p{p}_{ds}_r{r}.json"
                if out.exists() and not args.force:
                    continue
                cmd = mpirun_cmd(p) + [str(BUILD / "pagerank_mpi"), "--graph", str(graph),
                                       "--mode", "alltoallv", "--partition", "edge", "--iters", "30"]
                run_one(cmd, out, args.timeout)


def system_info():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    lscpu = sh("lscpu")
    field = lambda name: (re.search(rf"^{name}:\s*(.+)$", lscpu, re.M) or [None, ""])[1].strip()
    mem_kb = re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else "")
    info = {
        "cpu": field("Model name"),
        "logical_cpus": os.cpu_count(),
        "cores": int(field("Core\\(s\\) per socket") or 0) * int(field("Socket\\(s\\)") or 1),
        "threads_per_core": field("Thread\\(s\\) per core"),
        "l2": field("L2 cache"),
        "l3": field("L3 cache"),
        "memory_gb": round(int(mem_kb.group(1)) / 1024 / 1024, 1) if mem_kb else None,
        "os": sh(". /etc/os-release && echo $PRETTY_NAME"),
        "kernel": platform.release(),
        "mpi": sh("mpirun --version | head -1"),
        "compiler": sh("g++ --version | head -1"),
        "flags": "-O3 -march=native -std=c++17",
        "binding": "one rank per physical core (--bind-to core); P > cores uses hyper-threads",
    }
    (RESULTS / "system.json").write_text(json.dumps(info, indent=2))
    return info


def load_runs():
    rows = []
    for path in sorted(RUNS.glob("*/*.json")):
        d = json.loads(path.read_text())
        rows.append({
            "kind": "weak" if path.parent.name == "weak" else "strong",
            "dataset": d["graph"], "program": d["program"], "procs": d["procs"],
            "mode": d["mode"], "partition": d["partition"], "repeat": int(path.stem.rsplit("_r", 1)[1]),
            "n": d["n"], "m": d["m"], "iterations": d["iterations"], "converged": d["converged"],
            "t_load": d["t_load"], "t_setup": d["t_setup"], "t_solve": d["t_solve"], "t_total": d["t_total"],
            "t_compute_max": d["t_compute_max"], "t_compute_avg": d["t_compute_avg"],
            "t_exchange_max": d["t_exchange_max"], "t_exchange_avg": d["t_exchange_avg"],
            "t_reduce_max": d["t_reduce_max"], "t_reduce_avg": d["t_reduce_avg"],
            "imbalance": d["imbalance"], "bytes_per_iter": d["bytes_per_iter"], "mteps": d["mteps"],
            "file": str(path.relative_to(RESULTS).as_posix()),
        })
    return rows


def aggregate():
    import pandas as pd

    rows = load_runs()
    if not rows:
        print("no runs found")
        return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "bench.csv", index=False)

    keys = ["kind", "dataset", "program", "procs", "mode", "partition"]
    num = [c for c in df.columns if c.startswith("t_")] + ["imbalance", "bytes_per_iter", "mteps", "iterations"]
    med = df.groupby(keys, as_index=False)[num].median()
    med["repeats"] = df.groupby(keys).size().values
    med["t_solve_std"] = df.groupby(keys)["t_solve"].std().fillna(0).values

    seq = med[med.program == "seq"].set_index("dataset")
    def speedup(row, col):
        if row.kind == "weak":
            return float("nan")
        return seq.loc[row.dataset, col] / row[col] if row.dataset in seq.index else float("nan")
    med["speedup"] = med.apply(lambda r: speedup(r, "t_solve"), axis=1)
    med["speedup_total"] = med.apply(lambda r: speedup(r, "t_total"), axis=1)
    med["efficiency"] = med.speedup / med.procs
    # Karp-Flatt experimentally determined serial fraction: e = (1/S - 1/p) / (1 - 1/p)
    med["karp_flatt"] = med.apply(
        lambda r: (1 / r.speedup - 1 / r.procs) / (1 - 1 / r.procs) if r.procs > 1 and r.speedup > 0 else float("nan"),
        axis=1)

    weak = med[med.kind == "weak"].sort_values("procs")
    if not weak.empty:
        base = weak.iloc[0].t_solve
        med.loc[weak.index, "efficiency"] = base / weak.t_solve
    med.to_csv(RESULTS / "summary.csv", index=False)
    print(f"wrote {RESULTS / 'bench.csv'} ({len(df)} runs) and {RESULTS / 'summary.csv'} ({len(med)} configs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sweep on web-Google")
    ap.add_argument("--aggregate", action="store_true", help="only aggregate existing runs")
    ap.add_argument("--datasets", help="comma-separated dataset names (files in $DATA_DIR)")
    ap.add_argument("--procs", help="comma-separated process counts")
    ap.add_argument("--modes")
    ap.add_argument("--partitions")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--iters", type=int, help="fixed iteration count instead of convergence")
    ap.add_argument("--no-weak", dest="weak", action="store_false")
    ap.add_argument("--force", action="store_true", help="re-run configurations that already have results")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.aggregate:
        if not (BUILD / "pagerank_mpi").exists():
            sys.exit("build first: make")
        print(json.dumps(system_info(), indent=2))
        sweep(args)
    aggregate()


if __name__ == "__main__":
    main()
