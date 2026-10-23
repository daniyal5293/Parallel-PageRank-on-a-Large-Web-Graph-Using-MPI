"""Compares two PageRank runs: rank vectors (L1) and, optionally, top-k lists.

usage: compare_ranks.py <a.f64> <b.f64> [--max-l1 1e-12] [--json-a a.json --json-b b.json]
"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--max-l1", type=float, default=1e-12)
    ap.add_argument("--json-a")
    ap.add_argument("--json-b")
    args = ap.parse_args()

    a = np.fromfile(args.a, dtype="<f8")
    b = np.fromfile(args.b, dtype="<f8")
    if a.size != b.size:
        sys.exit(f"size mismatch: {a.size} vs {b.size}")
    l1 = float(np.abs(a - b).sum())
    ok = l1 <= args.max_l1
    msg = f"L1 = {l1:.3e}"

    if args.json_a and args.json_b:
        ta = json.load(open(args.json_a))["top"]
        tb = json.load(open(args.json_b))["top"]
        scores_match = len(ta) == len(tb) and all(abs(x["score"] - y["score"]) <= 1e-12 for x, y in zip(ta, tb))
        # Ids must agree wherever scores are not tied with a neighbour.
        tied = {i for i in range(len(ta) - 1) if abs(ta[i]["score"] - ta[i + 1]["score"]) <= 1e-12}
        tied |= {i + 1 for i in tied}
        ids_match = all(x["id"] == y["id"] for i, (x, y) in enumerate(zip(ta, tb)) if i not in tied)
        ok = ok and scores_match and ids_match
        msg += f", top-{len(ta)} {'match' if scores_match and ids_match else 'DIFFER'}"

    print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
