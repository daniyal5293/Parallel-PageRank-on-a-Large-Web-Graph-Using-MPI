// platform_probe: measures how far the machine itself lets P processes scale.
//
//   mpirun -np P ./build/platform_probe [--json out.json]
//
// Three fixed-size workloads are split evenly over the P processes (strong
// scaling, like the PageRank runs), and their speedup over P = 1 shows the
// hardware ceiling that any parallel program on this machine runs into:
//   compute : independent multiply-add chains, no memory traffic
//             -> limited by core count and all-core clock / power budget
//   stream  : STREAM-style triad a[i] = b[i] + s * c[i] over 3 x 128 MB
//             -> limited by shared memory bandwidth
//   gather  : sum += table[idx[i]] with a 32 MB table per process and a
//             streamed index array -> the access pattern of the PageRank pull
//             loop (in_src streamed, contributions read at random)
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "json.hpp"

using namespace pr;

namespace {

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Runs `body` after a barrier and returns the slowest process's time.
template <class F>
double timed(F body, MPI_Comm comm) {
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    body();
    double t = MPI_Wtime() - t0, t_max = 0;
    MPI_Allreduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return t_max;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    Args a(argc, argv);
    volatile double sink = 0;

    // compute: 8 independent chains so the FMA units, not latency, set the pace.
    const uint64_t total_ops = 3200000000ULL;
    const uint64_t my_ops = total_ops / P / 8;
    const double t_compute = timed([&] {
        double x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        for (uint64_t i = 0; i < my_ops; ++i)
            for (double& v : x) v = v * 0.999999 + 1e-7;
        sink = x[0] + x[7];
    }, MPI_COMM_WORLD);

    // stream: 3 arrays of 128 MB in total, split over the processes, 10 sweeps.
    const size_t n = (128ULL << 20) / sizeof(double) / P;
    std::vector<double> va(n, 0.0), vb(n, 1.0), vc(n, 2.0);
    const int sweeps = 10;
    const double t_stream = timed([&] {
        for (int s = 0; s < sweeps; ++s)
            for (size_t i = 0; i < n; ++i) va[i] = vb[i] + 3.0 * vc[i];
        sink = va[n / 2];
    }, MPI_COMM_WORLD);
    const double stream_gbs = 3.0 * 8.0 * double(n) * P * sweeps / t_stream / 1e9;

    // gather: 32 MB table per process (4x the L3 cache), 400 M random reads in total.
    const size_t table_n = (32ULL << 20) / sizeof(double);
    const size_t reads = 400000000ULL / P;
    std::vector<double> table(table_n, 1.0);
    std::vector<uint32_t> idx(1 << 22);
    uint64_t seed = 12345 + uint64_t(rank);
    for (uint32_t& i : idx) i = uint32_t(splitmix(seed) % table_n);
    const double t_gather = timed([&] {
        double sum = 0;
        for (size_t r = 0; r < reads; ++r) sum += table[idx[r & (idx.size() - 1)]];
        sink = sum;
    }, MPI_COMM_WORLD);
    const double gather_rate = 400000000.0 / t_gather / 1e6;

    if (rank == 0) {
        std::printf("P=%d  compute %.3fs  stream %.3fs (%.1f GB/s)  gather %.3fs (%.0f M reads/s)\n", P, t_compute,
                    t_stream, stream_gbs, t_gather, gather_rate);
        if (a.has("json")) {
            JsonObject j;
            j.num("procs", P).num("t_compute", t_compute).num("t_stream", t_stream).num("stream_gbs", stream_gbs)
                .num("t_gather", t_gather).num("gather_mreads", gather_rate);
            write_text(a.get("json"), j.dump());
        }
    }
    MPI_Finalize();
    return 0;
}
