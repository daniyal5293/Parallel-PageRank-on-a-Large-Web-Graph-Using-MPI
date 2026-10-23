"""Checks a PageRank vector against networkx.pagerank on the same graph.

usage: check_networkx.py <graph.bin> <ranks.f64> [--damping 0.85] [--max-l1 1e-10]
"""
import argparse
import sys

import networkx as nx
import numpy as np


def read_graph(path):
    """Reads the PRCSR1 binary CSR format (see include/graph.hpp)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"PRCSR1"):
            sys.exit(f"{path}: not a PRCSR1 graph")
        n, m = np.frombuffer(f.read(16), dtype="<u8")
        in_off = np.frombuffer(f.read(8 * (int(n) + 1)), dtype="<u8")
        in_src = np.frombuffer(f.read(4 * int(m)), dtype="<u4")
    return int(n), in_off, in_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("ranks")
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--max-l1", type=float, default=1e-10)
    args = ap.parse_args()

    n, in_off, in_src = read_graph(args.graph)
    dst = np.repeat(np.arange(n), np.diff(in_off).astype(np.int64))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(in_src.tolist(), dst.tolist()))

    expected = nx.pagerank(G, alpha=args.damping, tol=1e-15, max_iter=100000)
    expected = np.array([expected[v] for v in range(n)])
    got = np.fromfile(args.ranks, dtype="<f8")
    if got.size != n:
        sys.exit(f"rank vector has {got.size} entries, graph has {n} vertices")

    l1 = float(np.abs(got - expected).sum())
    print(f"n={n} m={in_src.size}  L1(ours - networkx) = {l1:.3e}")
    sys.exit(0 if l1 <= args.max_l1 else 1)


if __name__ == "__main__":
    main()
