// pagerank_seq: sequential PageRank baseline.
//
//   ./build/pagerank_seq --graph web-Google.bin [--tol 1e-9] [--max-iter 200]
//                        [--iters K] [--damping 0.85] [--topk 10]
//                        [--out ranks.f64] [--json run.json] [--verbose]
//
// Uses exactly the same pull-based kernel and edge order as pagerank_mpi, so
// the two produce matching results and the speedup comparison is fair.
#include <cmath>
#include <cstdio>
#include <vector>

#include "cli.hpp"
#include "graph.hpp"
#include "json.hpp"
#include "pagerank.hpp"
#include "timer.hpp"

using namespace pr;

int main(int argc, char** argv) {
    try {
        Args a(argc, argv);
        if (!a.has("graph")) {
            std::fprintf(stderr, "usage: pagerank_seq --graph <file.bin> [--tol T] [--iters K] [--out f] [--json f]\n");
            return 2;
        }
        const std::string path = a.get("graph");
        const Params p = params_from_args(a);
        const bool verbose = a.has("verbose");

        double t_start = wall_time();
        Graph g = read_graph(path);
        double t_load = wall_time() - t_start;

        const uint64_t n = g.n;
        const double d = p.damping;
        std::vector<double> inv_deg(n), r(n, 1.0 / double(n)), r_new(n), contrib(n);
        double dangling = 0.0;
        for (uint64_t v = 0; v < n; ++v) {
            inv_deg[v] = g.out_deg[v] ? 1.0 / g.out_deg[v] : 0.0;
            if (g.out_deg[v] == 0) dangling += r[v];
        }

        std::vector<double> residuals;
        int iters = 0;
        bool converged = false;
        double t_solve_start = wall_time();
        for (int it = 0; it < p.iteration_limit(); ++it) {
            for (uint64_t u = 0; u < n; ++u) contrib[u] = r[u] * inv_deg[u];

            const double base = (1.0 - d) / double(n) + d * dangling / double(n);
            double diff = 0.0, next_dangling = 0.0;
            for (uint64_t v = 0; v < n; ++v) {
                double sum = 0.0;
                for (uint64_t e = g.in_off[v]; e < g.in_off[v + 1]; ++e) sum += contrib[g.in_src[e]];
                const double x = base + d * sum;
                diff += std::fabs(x - r[v]);
                if (g.out_deg[v] == 0) next_dangling += x;
                r_new[v] = x;
            }
            r.swap(r_new);
            dangling = next_dangling;
            residuals.push_back(diff);
            iters = it + 1;
            if (verbose) std::printf("  iter %3d  residual %.3e\n", iters, diff);
            if (diff < p.tol) converged = true;
            if (p.should_stop(diff)) break;
        }
        double t_solve = wall_time() - t_solve_start;

        double rank_sum = 0.0;
        for (double x : r) rank_sum += x;
        auto top = top_k(r.data(), n, 0, p.topk);
        IdMap ids(path);
        double t_total = wall_time() - t_start;

        if (a.has("out")) {
            FILE* f = std::fopen(a.get("out").c_str(), "wb");
            if (!f) die("cannot create " + a.get("out"));
            fwrite_all(f, r.data(), r.size(), a.get("out"));
            std::fclose(f);
        }
        const double residual = residuals.empty() ? 0.0 : residuals.back();
        const double mteps = t_solve > 0 ? double(g.m) * iters / t_solve / 1e6 : 0.0;

        std::printf("Sequential PageRank | graph=%s  n=%llu  m=%llu\n", basename_of(path).c_str(),
                    (unsigned long long)n, (unsigned long long)g.m);
        std::printf("%s after %d iterations (L1 residual %.3e, rank sum %.12f)\n",
                    converged ? "converged" : "stopped", iters, residual, rank_sum);
        std::printf("time: load %.3fs | solve %.3fs (%.2f ms/iter, %.1f MTEPS) | total %.3fs\n", t_load, t_solve,
                    1e3 * t_solve / std::max(iters, 1), mteps, t_total);
        print_top_k(top, ids);

        if (a.has("json")) {
            unsigned long long n_dangling = 0, max_in = 0, max_out = 0;
            for (uint64_t v = 0; v < n; ++v) {
                n_dangling += (g.out_deg[v] == 0);
                max_in = std::max<unsigned long long>(max_in, g.in_off[v + 1] - g.in_off[v]);
                max_out = std::max<unsigned long long>(max_out, g.out_deg[v]);
            }
            JsonObject j;
            j.str("program", "seq").str("graph", basename_of(path)).num("procs", 1)
                .str("mode", "seq").str("partition", "none")
                .num("n", (unsigned long long)n).num("m", (unsigned long long)g.m)
                .num("dangling", n_dangling).num("max_in_deg", max_in).num("max_out_deg", max_out)
                .num("damping", d).num("tol", p.tol)
                .num("iterations", iters).boolean("converged", converged).num("final_residual", residual)
                .num("rank_sum", rank_sum)
                .num("t_load", t_load).num("t_setup", 0.0).num("t_solve", t_solve).num("t_total", t_total)
                .num("t_compute_max", t_solve).num("t_compute_avg", t_solve)
                .num("t_exchange_max", 0.0).num("t_exchange_avg", 0.0)
                .num("t_reduce_max", 0.0).num("t_reduce_avg", 0.0)
                .num("imbalance", 1.0).num("bytes_per_iter", 0.0).num("mteps", mteps)
                .arr("residuals", residuals)
                .raw("top", top_k_json(top, ids));
            write_text(a.get("json"), j.dump());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "pagerank_seq: %s\n", e.what());
        return 1;
    }
}
