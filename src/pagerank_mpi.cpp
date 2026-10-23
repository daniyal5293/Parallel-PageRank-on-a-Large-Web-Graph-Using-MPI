// pagerank_mpi: distributed-memory PageRank with MPI.
//
//   mpirun -np 4 ./build/pagerank_mpi --graph web-Google.bin
//          [--mode allgather|alltoallv|overlap] [--partition vertex|edge] [--vertex-weight 4]
//          [--tol 1e-9] [--max-iter 200] [--iters K] [--damping 0.85] [--topk 10]
//          [--out ranks.f64] [--json run.json] [--verbose]
//
// Design
// ------
// * 1D block partitioning: rank q owns the contiguous vertex range
//   [bounds[q], bounds[q+1]) and all in-edges of those vertices, so it is the
//   only writer of their ranks (pull-based update, no write conflicts).
//     --partition vertex : equal number of vertices per rank (naive)
//     --partition edge   : equal (in-edges + w * vertices) per rank, w = 4 by
//                          default (--vertex-weight, calibrated with
//                          scripts/calibrate_partition.sh); boundaries found
//                          by binary search over in_off with MPI-IO
// * Parallel input: every rank reads only its own slice of the CSR arrays with
//   MPI_File_read_at. No rank ever holds the whole graph.
// * Each iteration, the owner of u publishes contrib[u] = r[u] / outdeg[u];
//   rank q then needs contrib[] of every source of its in-edges.
//     --mode allgather : MPI_Allgatherv of the whole contrib vector (O(N) per rank)
//     --mode alltoallv : ghost exchange. In setup each rank tells the owners
//                        which remote vertices it needs; then every iteration
//                        MPI_Alltoallv moves only those values (O(ghosts)).
//     --mode overlap   : as alltoallv, but MPI_Ialltoallv runs while in-edges
//                        with local sources are processed; ghost-source edges
//                        are processed after MPI_Wait.
// * The dangling mass and the L1 residual are combined in one MPI_Allreduce.
// * Top-k: every rank computes a local top-k; rank 0 merges the k*P candidates.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "graph.hpp"
#include "json.hpp"
#include "pagerank.hpp"

using namespace pr;

namespace {

enum class Mode { Allgather, Alltoallv, Overlap };

Mode parse_mode(const std::string& s) {
    if (s == "allgather") return Mode::Allgather;
    if (s == "alltoallv") return Mode::Alltoallv;
    if (s == "overlap") return Mode::Overlap;
    die("unknown --mode '" + s + "' (expected allgather, alltoallv or overlap)");
}

// Reads `bytes` bytes at `offset` with independent MPI-IO, in chunks that fit
// the int count argument.
void read_at(MPI_File fh, uint64_t offset, void* buf, uint64_t bytes) {
    char* p = static_cast<char*>(buf);
    while (bytes > 0) {
        const int chunk = int(std::min<uint64_t>(bytes, 1ULL << 30));
        MPI_Status st;
        MPI_File_read_at(fh, MPI_Offset(offset), p, chunk, MPI_BYTE, &st);
        int got = 0;
        MPI_Get_count(&st, MPI_BYTE, &got);
        if (got != chunk) die("short read from graph file");
        offset += uint64_t(chunk);
        p += chunk;
        bytes -= uint64_t(chunk);
    }
}

uint64_t read_u64_at(MPI_File fh, uint64_t offset) {
    uint64_t x;
    read_at(fh, offset, &x, sizeof(x));
    return x;
}

std::vector<uint64_t> vertex_partition(uint64_t n, int P) {
    std::vector<uint64_t> bounds(P + 1);
    for (int q = 0; q <= P; ++q) bounds[q] = n * uint64_t(q) / uint64_t(P);
    return bounds;
}

// Each rank q > 0 binary-searches the smallest v with
// in_off[v] + w * v >= q/P * (m + w * n), reading single in_off entries from
// the file, then the boundaries are shared with MPI_Allgather.
std::vector<uint64_t> edge_partition(MPI_File fh, const Header& h, int P, int rank, double w, MPI_Comm comm) {
    uint64_t my_bound = 0;
    if (rank > 0) {
        const double target = (double(h.m) + w * double(h.n)) * rank / P;
        uint64_t lo = 0, hi = h.n;
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2;
            const double cost = double(read_u64_at(fh, in_off_pos(h) + 8 * mid)) + w * double(mid);
            if (cost >= target) hi = mid;
            else lo = mid + 1;
        }
        my_bound = lo;
    }
    std::vector<uint64_t> bounds(P + 1);
    MPI_Allgather(&my_bound, 1, MPI_UINT64_T, bounds.data(), 1, MPI_UINT64_T, comm);
    bounds[P] = h.n;
    return bounds;
}

std::vector<int> prefix_displs(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    for (size_t i = 1; i < counts.size(); ++i) displs[i] = displs[i - 1] + counts[i - 1];
    return displs;
}

template <class T>
double max_of(const std::vector<T>& v) { return v.empty() ? 0.0 : double(*std::max_element(v.begin(), v.end())); }
template <class T>
double avg_of(const std::vector<T>& v) {
    double s = 0;
    for (T x : v) s += double(x);
    return v.empty() ? 0.0 : s / double(v.size());
}

// Per-rank statistics gathered on rank 0 (order matters, see kStatNames).
enum Stat { kVertices, kEdges, kGhosts, kServed, kRecvValues, kLoad, kSetup, kCompute, kExchange, kReduce, kSolve, kNumStats };
const char* const kStatNames[kNumStats] = {"vertices", "edges", "ghosts", "served", "recv_values", "t_load",
                                           "t_setup", "t_compute", "t_exchange", "t_reduce", "t_solve"};

int run(int argc, char** argv) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    Args a(argc, argv);
    if (!a.has("graph")) {
        if (rank == 0)
            std::fprintf(stderr, "usage: pagerank_mpi --graph <file.bin> [--mode allgather|alltoallv|overlap] "
                                 "[--partition vertex|edge] [--tol T] [--iters K] [--out f] [--json f]\n");
        return 2;
    }
    const std::string path = a.get("graph");
    const std::string mode_name = a.get("mode", "alltoallv");
    const Mode mode = parse_mode(mode_name);
    const std::string partition = a.get("partition", "edge");
    if (partition != "vertex" && partition != "edge") die("unknown --partition '" + partition + "'");
    const Params p = params_from_args(a);
    const bool verbose = a.has("verbose");

    MPI_Barrier(comm);
    const double t_start = MPI_Wtime();

    // ---------------------------------------------------------------- load
    MPI_File fh;
    if (MPI_File_open(comm, path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
        die("cannot open " + path);
    Header h;
    if (rank == 0) read_at(fh, 0, &h, sizeof(h));
    MPI_Bcast(&h, sizeof(h), MPI_BYTE, 0, comm);
    check_header(h, path);
    const uint64_t n = h.n;

    const std::vector<uint64_t> bounds = partition == "edge"
        ? edge_partition(fh, h, P, rank, a.get_double("vertex-weight", 4.0), comm)
        : vertex_partition(n, P);
    const uint64_t lo = bounds[rank], hi = bounds[rank + 1], nloc = hi - lo;

    std::vector<uint64_t> off(nloc + 1);
    read_at(fh, in_off_pos(h) + 8 * lo, off.data(), 8 * (nloc + 1));
    const uint64_t e0 = off[0], mloc = off[nloc] - e0;
    for (uint64_t& o : off) o -= e0;
    std::vector<uint32_t> col(mloc), out_deg(nloc);
    read_at(fh, in_src_pos(h) + 4 * e0, col.data(), 4 * mloc);
    read_at(fh, out_deg_pos(h) + 4 * lo, out_deg.data(), 4 * nloc);
    MPI_File_close(&fh);
    const double t_load = MPI_Wtime() - t_start;

    // --------------------------------------------------------------- setup
    const double t_setup_start = MPI_Wtime();
    std::vector<double> x;      // contributions: allgather -> global [0,n); ghost modes -> [local | ghost]
    std::vector<int> ag_counts, ag_displs;                                      // allgather
    std::vector<int> need_counts(P, 0), need_displs, serve_counts(P), serve_displs;  // ghost exchange
    std::vector<uint32_t> serve_idx;   // local vertices whose contribution other ranks need
    std::vector<double> send_buf;
    std::vector<uint64_t> loc_off, gh_off;  // overlap: split CSR (local-source / ghost-source edges)
    std::vector<uint32_t> loc_col, gh_col;
    uint64_t nghost = 0;

    if (mode == Mode::Allgather) {
        if (n > uint64_t(INT32_MAX)) die("allgather mode needs n < 2^31");
        ag_counts.resize(P);
        for (int q = 0; q < P; ++q) ag_counts[q] = int(bounds[q + 1] - bounds[q]);
        ag_displs = prefix_displs(ag_counts);
        x.assign(n, 0.0);
    } else {
        // Ghosts: remote sources of local in-edges. A temporary global-id ->
        // ghost-slot table finds them in O(n + m) and lists them in increasing
        // id order, hence grouped by owner.
        constexpr uint32_t kUnused = UINT32_MAX;
        std::vector<uint32_t> slot(n, kUnused);
        for (uint32_t s : col)
            if (s < lo || s >= hi) slot[s] = 0;
        std::vector<uint32_t> ghosts;
        for (uint64_t u = 0; u < n; ++u)
            if (slot[u] != kUnused) {
                slot[u] = uint32_t(ghosts.size());
                ghosts.push_back(uint32_t(u));
            }
        nghost = ghosts.size();

        int q = 0;
        for (uint32_t g : ghosts) {
            while (g >= bounds[q + 1]) ++q;
            need_counts[q]++;
        }
        need_displs = prefix_displs(need_counts);
        MPI_Alltoall(need_counts.data(), 1, MPI_INT, serve_counts.data(), 1, MPI_INT, comm);
        serve_displs = prefix_displs(serve_counts);
        serve_idx.resize(size_t(serve_displs[P - 1]) + size_t(serve_counts[P - 1]));
        MPI_Alltoallv(ghosts.data(), need_counts.data(), need_displs.data(), MPI_UINT32_T, serve_idx.data(),
                      serve_counts.data(), serve_displs.data(), MPI_UINT32_T, comm);
        for (uint32_t& s : serve_idx) s -= uint32_t(lo);
        send_buf.resize(serve_idx.size());

        // Renumber edge sources into the compact [local | ghost] index space.
        for (uint32_t& s : col) s = (s >= lo && s < hi) ? s - uint32_t(lo) : uint32_t(nloc) + slot[s];
        std::vector<uint32_t>().swap(slot);
        x.assign(nloc + nghost, 0.0);

        if (mode == Mode::Overlap) {
            loc_off.assign(nloc + 1, 0);
            gh_off.assign(nloc + 1, 0);
            for (uint64_t v = 0; v < nloc; ++v) {
                for (uint64_t e = off[v]; e < off[v + 1]; ++e)
                    (col[e] < nloc ? loc_col : gh_col).push_back(col[e]);
                loc_off[v + 1] = loc_col.size();
                gh_off[v + 1] = gh_col.size();
            }
            std::vector<uint32_t>().swap(col);
        }
    }
    double* x_local = x.data() + (mode == Mode::Allgather ? lo : 0);

    const double d = p.damping;
    std::vector<double> inv_deg(nloc), r(nloc, 1.0 / double(n)), acc(mode == Mode::Overlap ? nloc : 0);
    double local_dangling = 0.0;
    for (uint64_t v = 0; v < nloc; ++v) {
        inv_deg[v] = out_deg[v] ? 1.0 / out_deg[v] : 0.0;
        if (out_deg[v] == 0) local_dangling += r[v];
    }
    double dangling = 0.0;
    MPI_Allreduce(&local_dangling, &dangling, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double t_setup = MPI_Wtime() - t_setup_start;

    // --------------------------------------------------------------- solve
    double t_compute = 0, t_exchange = 0, t_reduce = 0;
    std::vector<double> residuals;
    int iters = 0;
    bool converged = false;
    MPI_Barrier(comm);
    const double t_solve_start = MPI_Wtime();

    for (int it = 0; it < p.iteration_limit(); ++it) {
        const double t0 = MPI_Wtime();
        for (uint64_t v = 0; v < nloc; ++v) x_local[v] = r[v] * inv_deg[v];
        for (size_t i = 0; i < serve_idx.size(); ++i) send_buf[i] = x_local[serve_idx[i]];
        const double base = (1.0 - d) / double(n) + d * dangling / double(n);
        double partial[2] = {0.0, 0.0};  // {L1 residual, next dangling mass}

        auto finish_vertex = [&](uint64_t v, double sum) {
            const double y = base + d * sum;
            partial[0] += std::fabs(y - r[v]);
            if (out_deg[v] == 0) partial[1] += y;
            r[v] = y;  // safe: r[v] is no longer read once its own sum is known
        };

        double t1, t2;
        if (mode == Mode::Overlap) {
            MPI_Request req;
            MPI_Ialltoallv(send_buf.data(), serve_counts.data(), serve_displs.data(), MPI_DOUBLE, x.data() + nloc,
                           need_counts.data(), need_displs.data(), MPI_DOUBLE, comm, &req);
            int done = 0;
            for (uint64_t v = 0; v < nloc; ++v) {
                double sum = 0.0;
                for (uint64_t e = loc_off[v]; e < loc_off[v + 1]; ++e) sum += x[loc_col[e]];
                acc[v] = sum;
                if (!done && (v & 1023) == 1023) MPI_Test(&req, &done, MPI_STATUS_IGNORE);  // drive progress
            }
            t1 = MPI_Wtime();
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            t2 = MPI_Wtime();
            for (uint64_t v = 0; v < nloc; ++v) {
                double sum = acc[v];
                for (uint64_t e = gh_off[v]; e < gh_off[v + 1]; ++e) sum += x[gh_col[e]];
                finish_vertex(v, sum);
            }
        } else {
            t1 = MPI_Wtime();
            if (mode == Mode::Allgather)
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), ag_counts.data(), ag_displs.data(),
                               MPI_DOUBLE, comm);
            else
                MPI_Alltoallv(send_buf.data(), serve_counts.data(), serve_displs.data(), MPI_DOUBLE, x.data() + nloc,
                              need_counts.data(), need_displs.data(), MPI_DOUBLE, comm);
            t2 = MPI_Wtime();
            for (uint64_t v = 0; v < nloc; ++v) {
                double sum = 0.0;
                for (uint64_t e = off[v]; e < off[v + 1]; ++e) sum += x[col[e]];
                finish_vertex(v, sum);
            }
        }
        const double t3 = MPI_Wtime();
        double global[2];
        MPI_Allreduce(partial, global, 2, MPI_DOUBLE, MPI_SUM, comm);
        const double t4 = MPI_Wtime();

        t_compute += (t1 - t0) + (t3 - t2);
        t_exchange += t2 - t1;
        t_reduce += t4 - t3;
        dangling = global[1];
        residuals.push_back(global[0]);
        iters = it + 1;
        if (verbose && rank == 0) std::printf("  iter %3d  residual %.3e\n", iters, global[0]);
        if (global[0] < p.tol) converged = true;
        if (p.should_stop(global[0])) break;
    }
    const double t_solve = MPI_Wtime() - t_solve_start;

    // ------------------------------------------------------------- results
    const auto local_top = top_k(r.data(), nloc, lo, p.topk);
    std::vector<double> packed(2 * size_t(p.topk), -1.0), gathered(rank == 0 ? packed.size() * P : 0);
    for (size_t i = 0; i < local_top.size(); ++i) {
        packed[2 * i] = local_top[i].score;
        packed[2 * i + 1] = double(local_top[i].id);
    }
    MPI_Gather(packed.data(), int(packed.size()), MPI_DOUBLE, gathered.data(), int(packed.size()), MPI_DOUBLE, 0, comm);

    double local_sum = 0.0, rank_sum = 0.0;
    for (double y : r) local_sum += y;
    MPI_Reduce(&local_sum, &rank_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    const double t_total = MPI_Wtime() - t_start;

    if (a.has("out")) {  // collective parallel write: each rank writes its own slice
        const std::string out = a.get("out");
        if (rank == 0) MPI_File_delete(out.c_str(), MPI_INFO_NULL);
        MPI_Barrier(comm);
        MPI_File ofh;
        if (MPI_File_open(comm, out.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &ofh) != MPI_SUCCESS)
            die("cannot create " + out);
        MPI_File_write_at_all(ofh, MPI_Offset(lo * 8), r.data(), int(nloc), MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&ofh);
    }

    double stats[kNumStats] = {double(nloc), double(mloc), double(nghost), double(serve_idx.size()),
                               mode == Mode::Allgather ? double(n - nloc) : double(nghost),
                               t_load, t_setup, t_compute, t_exchange, t_reduce, t_solve};
    std::vector<double> all_stats(rank == 0 ? size_t(kNumStats) * P : 0);
    MPI_Gather(stats, kNumStats, MPI_DOUBLE, all_stats.data(), kNumStats, MPI_DOUBLE, 0, comm);
    if (rank != 0) return 0;

    // Rank 0: merge top-k, summarize per-rank statistics, report.
    std::vector<RankedNode> candidates;
    for (size_t i = 0; i < gathered.size(); i += 2)
        if (gathered[i] >= 0.0) candidates.push_back({gathered[i], uint64_t(gathered[i + 1])});
    const auto top = merge_top_k(candidates, p.topk);
    IdMap ids(path);

    std::vector<std::vector<double>> per(kNumStats, std::vector<double>(P));
    for (int q = 0; q < P; ++q)
        for (int s = 0; s < kNumStats; ++s) per[s][q] = all_stats[size_t(q) * kNumStats + s];
    double recv_total = 0;
    for (double y : per[kRecvValues]) recv_total += y;
    const double bytes_per_iter = 8.0 * recv_total;
    const double imbalance = avg_of(per[kCompute]) > 0 ? max_of(per[kCompute]) / avg_of(per[kCompute]) : 1.0;
    const double residual = residuals.empty() ? 0.0 : residuals.back();
    const double t_solve_max = max_of(per[kSolve]);
    const double mteps = t_solve_max > 0 ? double(h.m) * iters / t_solve_max / 1e6 : 0.0;

    std::printf("Parallel PageRank (MPI) | graph=%s  n=%llu  m=%llu\n", basename_of(path).c_str(),
                (unsigned long long)n, (unsigned long long)h.m);
    std::printf("processes=%d  mode=%s  partition=%s\n", P, mode_name.c_str(), partition.c_str());
    std::printf("%s after %d iterations (L1 residual %.3e, rank sum %.12f)\n", converged ? "converged" : "stopped",
                iters, residual, rank_sum);
    std::printf("time: load %.3fs | setup %.3fs | solve %.3fs (%.2f ms/iter, %.1f MTEPS) | total %.3fs\n",
                max_of(per[kLoad]), max_of(per[kSetup]), t_solve_max, 1e3 * t_solve_max / std::max(iters, 1), mteps,
                t_total);
    std::printf("communication: %.2f MB per iteration, compute imbalance (max/avg) %.3f\n", bytes_per_iter / 1e6,
                imbalance);
    if (P <= 16 || verbose) {
        std::printf("\n  %-4s %12s %12s %10s %10s %10s %10s %10s\n", "rank", "vertices", "in-edges", "ghosts",
                    "compute", "exchange", "reduce", "solve");
        for (int q = 0; q < P; ++q)
            std::printf("  %-4d %12.0f %12.0f %10.0f %9.3fs %9.3fs %9.3fs %9.3fs\n", q, per[kVertices][q],
                        per[kEdges][q], per[kGhosts][q], per[kCompute][q], per[kExchange][q], per[kReduce][q],
                        per[kSolve][q]);
    }
    print_top_k(top, ids);

    if (a.has("json")) {
        JsonObject per_rank;
        for (int s = 0; s < kNumStats; ++s) per_rank.arr(kStatNames[s], per[s]);
        JsonObject j;
        j.str("program", "mpi").str("graph", basename_of(path)).num("procs", P)
            .str("mode", mode_name).str("partition", partition)
            .num("n", (unsigned long long)n).num("m", (unsigned long long)h.m)
            .num("damping", d).num("tol", p.tol)
            .num("iterations", iters).boolean("converged", converged).num("final_residual", residual)
            .num("rank_sum", rank_sum)
            .num("t_load", max_of(per[kLoad])).num("t_setup", max_of(per[kSetup]))
            .num("t_solve", t_solve_max).num("t_total", t_total)
            .num("t_compute_max", max_of(per[kCompute])).num("t_compute_avg", avg_of(per[kCompute]))
            .num("t_exchange_max", max_of(per[kExchange])).num("t_exchange_avg", avg_of(per[kExchange]))
            .num("t_reduce_max", max_of(per[kReduce])).num("t_reduce_avg", avg_of(per[kReduce]))
            .num("imbalance", imbalance).num("bytes_per_iter", bytes_per_iter).num("mteps", mteps)
            .arr("bounds", bounds)
            .arr("residuals", residuals)
            .raw("per_rank", per_rank.dump())
            .raw("top", top_k_json(top, ids));
        write_text(a.get("json"), j.dump());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int code = 0;
    try {
        code = run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pagerank_mpi: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return code;
}
