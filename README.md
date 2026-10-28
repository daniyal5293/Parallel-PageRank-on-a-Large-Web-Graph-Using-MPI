# Parallel PageRank on Large Web Graphs (MPI)

PDC course project by **Daniyal Ahmed Khan (23K-3039), Abdul Hadi (23K-3041), Taha Ali Khan (23K-3005),
Muhammad Tahseen (23K-3040)**.

The project has a sequential and a distributed-memory (MPI) PageRank in C++. They run on real web and
social graphs with up to 4.8 M nodes and 69 M edges, and on synthetic R-MAT graphs. Benchmark scripts
measure execution time, speedup, efficiency, communication volume and load balance, and generate the
plots and an interactive dashboard.

## Layout

| Path | What it is |
|---|---|
| `src/pagerank_seq.cpp` | Sequential baseline (same kernel as the MPI version) |
| `src/pagerank_mpi.cpp` | MPI implementation: 2 partitioners × 3 communication strategies |
| `src/convert.cpp` | SNAP/KONECT edge list → binary CSR graph |
| `src/gen_rmat.cpp` | Synthetic power-law (R-MAT) graph generator |
| `src/reorder.cpp` | Vertex relabelling by out-degree (cache-locality experiment) |
| `src/platform_probe.cpp` | Hardware ceiling probe: compute / memory-stream / random-gather kernels under MPI |
| `include/` | Graph format, PageRank helpers, CLI/JSON utilities |
| `tests/` | Correctness suite (vs networkx, and MPI vs sequential) |
| `scripts/` | Setup, dataset download/prepare, benchmark, plotting, dashboard |
| `results/` | Benchmark CSVs, raw per-run JSON, plots |
| `dashboard/index.html` | Interactive results dashboard (open in a browser) |

## Algorithm

```
r'[v] = (1-d)/N + d * ( Σ_{u→v} r[u]/outdeg[u]  +  D/N )        d = 0.85
```

`D` is the total rank on dangling vertices (no out-links), redistributed uniformly. The iteration
stops when `‖r' − r‖₁ < tol` (default `1e-9`) or after `max_iter` iterations. The graph is stored
as a CSR of **in-edges** ("pull" form). Each vertex's new rank is written by exactly one process,
so no locks or atomics are needed.

### Parallel design (`pagerank_mpi`)

* **Partitioning.** 1D blocks of contiguous vertices.
  * `--partition vertex`: equal vertex counts per process.
  * `--partition edge` (default): equal `in-edges + w·vertices` per process. Each process
    binary-searches its boundary in the on-disk offset array (MPI-IO), and the boundaries are shared
    with `MPI_Allgather`. The weight `w = 4` (`--vertex-weight`) was calibrated with
    `scripts/calibrate_partition.sh`.
* **Parallel input.** Each process reads only its own slice of the graph with `MPI_File_read_at`.
* **Communication strategy** (`--mode`):
  * `allgather`: `MPI_Allgatherv` replicates the whole contribution vector `r[u]/outdeg[u]` on
    every process. Simple, but it moves O(N) data per process per iteration.
  * `alltoallv`: ghost exchange. During setup, each process tells the owners which remote vertices
    it needs (`MPI_Alltoall` + `MPI_Alltoallv`). Each iteration then sends only those values with
    `MPI_Alltoallv`. Edge sources are renumbered into a compact local index space for cache locality.
  * `overlap`: like `alltoallv`, but with non-blocking `MPI_Ialltoallv`. In-edges whose source is
    local are processed while the exchange is in flight.
* The dangling mass and the convergence residual are reduced together in one `MPI_Allreduce` per
  iteration.
* For the top-k, each process finds its local top-k and rank 0 merges the `k × P` candidates
  (`MPI_Gather`). The optional full rank vector is written with collective `MPI_File_write_at_all`.

## Setup (Windows + WSL)

```powershell
wsl --install -d Ubuntu-24.04          # once, then open the Ubuntu shell
```
```bash
cd /mnt/c/Users/<you>/Desktop/pdc-project
bash scripts/setup_wsl.sh              # g++, OpenMPI, Python venv (~/.venvs/pdc)
make                                   # builds build/{pagerank_seq,pagerank_mpi,convert,gen_rmat}
make test                              # 226 correctness checks
```

On any other Linux machine or cluster, the only requirements are `g++`, OpenMPI (or MPICH),
Python 3 and `make`.

## Datasets

```bash
bash scripts/download_datasets.sh      # web-Google, web-BerkStan, soc-LiveJournal1 (SNAP, KONECT mirror fallback)
bash scripts/prepare_datasets.sh       # -> ~/pdc-data/*.bin, plus R-MAT scales 20-23
```

| Graph | Vertices | Edges (after de-duplication) | Type |
|---|---|---|---|
| web-Google | 875,713 | 5,105,039 | web hyperlinks (Google, 2002) |
| web-BerkStan | 685,230 | 7,600,595 | web hyperlinks (berkeley.edu + stanford.edu) |
| soc-LiveJournal1 | 4,846,609 | 68,475,391 | social network (directed friendships) |
| rmat-20 … rmat-23 | 1.0 M … 8.4 M | 16.1 M … 130 M | synthetic power-law (Graph500 parameters) |

`snap.stanford.edu` is unreachable from some networks, so the download script falls back to the
[KONECT](http://konect.cc) mirror of the same graphs. KONECT renumbers vertices, which means the
reported node ids are KONECT ids.

Graph data is kept in `~/pdc-data` (set `DATA_DIR` to change it). That is inside the Linux
filesystem, because reading from `/mnt/c` is slow under WSL.

`build/reorder in.bin out.bin` relabels vertices by decreasing out-degree, as a cache-locality
experiment (`scripts/reorder_experiment.sh`). It made sequential web-Google 12% faster, but it made
4-process web-Google 23% slower and LiveJournal 44% slower in both cases. The benchmarks therefore
use the natural order.

## Running

```bash
./build/pagerank_seq --graph ~/pdc-data/web-Google.bin
mpirun -np 4 ./build/pagerank_mpi --graph ~/pdc-data/web-Google.bin --mode alltoallv --partition edge
```

Common options: `--tol 1e-9`, `--max-iter 200`, `--iters K` (fixed iteration count), `--damping 0.85`,
`--topk 10`, `--out ranks.f64` (binary float64 vector), `--json run.json` (machine-readable
summary) and `--verbose` (prints the residual every iteration).

## Viva demo and study guide

* `bash scripts/demo.sh` runs a one-minute live demo: the sequential program, then all three MPI
  strategies with 4 processes, then vertex vs edge partitioning. It writes nothing to `results/`.
* [PROJECT_GUIDE.txt](PROJECT_GUIDE.txt) explains the whole project from zero: concepts,
  architecture, every file, how to run it, results and likely viva questions.

## Reproducing the results

Run the benchmarks **on AC power with the lid open**. On battery the CPU is throttled, and standby
pauses the WSL VM in the middle of runs.

```bash
RESULTS_DIR=results-check bash scripts/run_all.sh   # fresh full run into its own folder (~80 min)
```

The `RESULTS_DIR` folder keeps a check run separate: `results/` and `dashboard/index.html` stay
untouched, and the new dashboard goes to `results-check/dashboard.html`. The pipeline is safe to
interrupt. Running the same command again skips every finished run. The dashboard is only replaced
in the last step, and only by a complete file.

The individual steps are:

```bash
source ~/.venvs/pdc/bin/activate
python scripts/bench.py                # benchmark sweep (resumable) -> results/bench.csv, summary.csv, runs/
bash scripts/calibrate_partition.sh    # vertex-weight calibration -> results/calibration.csv
bash scripts/probe_platform.sh         # hardware ceiling (compute / stream / gather) -> results/platform.csv
bash scripts/reorder_experiment.sh     # degree-order locality experiment -> results/reorder.csv
python scripts/plot_results.py         # results/plots/*.png
python scripts/build_dashboard.py      # dashboard/index.html
```

`python scripts/bench.py --quick` runs a few-minute sanity sweep on web-Google only.

## Results at a glance

These are speedups over the sequential program on an i5-1145G7 laptop (4 cores / 8 threads, WSL2,
Open MPI 4.1.6), using the median of 3 runs and the best strategy, edge-balanced partition:

| Graph | Sequential solve | Best speedup | Speedup at 4 processes |
|---|---|---|---|
| web-Google (5.1 M edges) | 1.45 s | 1.33× (8 processes) | 1.17× |
| web-BerkStan (7.6 M edges) | 1.13 s | 1.91× (6 processes) | 1.79× |
| soc-LiveJournal1 (68.5 M edges) | 23.16 s | 1.36× (2 processes) | 1.29× |
| rmat-22 (65.2 M edges) | 24.85 s | 4.11× (6 processes) | 4.00× |

The ghost exchange (alltoallv) moves 64–90% less data than Allgatherv. The platform probe shows
that one core already uses nearly all of the laptop's memory bandwidth (about 13 GB/s), and that
limit sets PageRank's speedup on the real graphs. The full analysis is in the project report.
