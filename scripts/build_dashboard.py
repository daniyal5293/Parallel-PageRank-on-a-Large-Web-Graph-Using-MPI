"""Builds the interactive results dashboard from the benchmark results.

  python scripts/build_dashboard.py          -> dashboard/index.html (open in any browser)
  python scripts/build_dashboard.py --fragment out.html
                                             -> page body only (for publishing as an artifact)

Reads results/summary.csv, results/system.json, results/runs/*/*.json and
results/tests.log (output of `make test`, optional).
"""
import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DASH = ROOT / "dashboard"
DATASET_ORDER = ["web-Google", "web-BerkStan", "soc-LiveJournal1", "rmat-22"]


def clean(x):
    """JSON-safe value (NaN -> None, numpy scalars -> Python)."""
    if isinstance(x, float) and math.isnan(x):
        return None
    return x.item() if hasattr(x, "item") else x


def collect():
    summary = pd.read_csv(RESULTS / "summary.csv")
    strong = summary[summary.kind == "strong"]
    seq = strong[strong.program == "seq"].set_index("dataset")
    mpi = strong[strong.program == "mpi"]

    names = [d for d in DATASET_ORDER if d in seq.index] + sorted(set(seq.index) - set(DATASET_ORDER))
    datasets, convergence, top = [], {}, {}
    for name in names:
        run = json.loads((RESULTS / "runs" / name / "seq_r0.json").read_text())
        datasets.append({
            "name": name, "n": run["n"], "m": run["m"], "dangling": run.get("dangling"),
            "max_in_deg": run.get("max_in_deg"), "max_out_deg": run.get("max_out_deg"),
            "iterations": run["iterations"], "t_seq": clean(seq.loc[name, "t_solve"]),
            "mteps_seq": clean(seq.loc[name, "mteps"]),
        })
        convergence[name] = run["residuals"]
        top[name] = [{"orig": t["orig"], "score": t["score"]} for t in run["top"]]

    cols = ["dataset", "procs", "mode", "partition", "t_solve", "t_solve_std", "t_total", "speedup", "efficiency",
            "karp_flatt", "t_compute_avg", "t_exchange_avg", "t_reduce_avg", "imbalance", "bytes_per_iter", "mteps"]
    rows = [{c: clean(r[c]) for c in cols} for _, r in mpi.iterrows()]

    per_rank = {}
    for f in sorted((RESULTS / "runs").glob("*/mpi_p*_r0.json")):
        if f.parent.name == "weak":
            continue
        d = json.loads(f.read_text())
        pr = d["per_rank"]
        per_rank[f"{d['graph']}|{d['procs']}|{d['mode']}|{d['partition']}"] = {
            k: pr[k] for k in ["t_compute", "t_exchange", "t_reduce", "edges", "vertices", "ghosts"]}

    weak = summary[summary.kind == "weak"].sort_values("procs")
    weak_rows = [{"procs": int(r.procs), "dataset": r.dataset, "t_solve": r.t_solve, "efficiency": clean(r.efficiency),
                  "m": None} for _, r in weak.iterrows()]
    for w in weak_rows:
        f = next((RESULTS / "runs" / "weak").glob(f"mpi_p{w['procs']}_{w['dataset']}_r0.json"), None)
        if f:
            w["m"] = json.loads(f.read_text())["m"]

    tests = {"passed": 0, "total": 0, "nx_l1": "n/a"}
    log = RESULTS / "tests.log"
    if log.exists():
        text = log.read_text()
        m = re.search(r"passed:\s*(\d+)\s+failed:\s*(\d+)", text)
        if m:
            tests["passed"], tests["total"] = int(m.group(1)), int(m.group(1)) + int(m.group(2))
        l1 = [float(x) for x in re.findall(r"L1\(ours - networkx\) = ([0-9.e+-]+)", text)]
        if l1:
            tests["nx_l1"] = f"{max(l1):.0e}"

    platform = []
    if (RESULTS / "platform.csv").exists():
        probe = pd.read_csv(RESULTS / "platform.csv").groupby("procs", as_index=False).median()
        base = probe[probe.procs == 1].iloc[0]
        platform = [{"procs": int(r.procs), "compute": base.t_compute / r.t_compute, "gather": base.t_gather / r.t_gather,
                     "stream": base.t_stream / r.t_stream, "stream_gbs": r.stream_gbs} for _, r in probe.iterrows()]

    return {
        "generated": dt.date.today().isoformat(),
        "platform": platform,
        "repeats": int(mpi.repeats.max()) if "repeats" in mpi and len(mpi) else 1,
        "system": json.loads((RESULTS / "system.json").read_text()),
        "datasets": datasets, "summary": rows, "perRank": per_rank, "convergence": convergence, "top": top,
        "weak": weak_rows, "tests": tests,
        "seq": {n: {"t_solve": clean(seq.loc[n, "t_solve"]), "t_total": clean(seq.loc[n, "t_total"])} for n in names},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="write only the page body (artifact publishing) to this path")
    args = ap.parse_args()

    data = json.dumps(collect(), separators=(",", ":"))
    body = (DASH / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", data)
    if args.fragment:
        Path(args.fragment).write_text(body, encoding="utf-8")
        print("wrote", args.fragment)
        return
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    out = DASH / "index.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
