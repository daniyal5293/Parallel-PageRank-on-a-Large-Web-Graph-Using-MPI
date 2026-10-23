// gen_rmat: synthetic power-law graph generator (R-MAT, Chakrabarti et al. 2004).
//
//   ./build/gen_rmat rmat-20.bin --scale 20 [--edgefactor 16] [--seed 1]
//                    [--a 0.57 --b 0.19 --c 0.19]
//
// Generates N = 2^scale vertices and edgefactor * N directed edges using the
// Graph500 quadrant probabilities, then applies a random vertex permutation
// so that high-degree vertices are not clustered at low ids (otherwise block
// partitions would be artificially unbalanced). Duplicate edges are removed,
// so the final edge count is slightly below edgefactor * N. Output is
// deterministic for a given seed.
#include <cinttypes>
#include <cstdio>
#include <numeric>
#include <vector>

#include "cli.hpp"
#include "graph.hpp"
#include "timer.hpp"

using namespace pr;

struct SplitMix64 {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return double(next() >> 11) * 0x1.0p-53; }
};

int main(int argc, char** argv) {
    try {
        Args a(argc, argv);
        if (a.positional().size() != 1 || !a.has("scale")) {
            std::fprintf(stderr, "usage: gen_rmat <out.bin> --scale S [--edgefactor 16] [--seed 1] [--a --b --c]\n");
            return 2;
        }
        const std::string out = a.positional()[0];
        const int scale = int(a.get_int("scale", 16));
        if (scale < 1 || scale > 31) die("scale must be in [1, 31]");
        const uint64_t n = 1ULL << scale;
        const uint64_t target = uint64_t(a.get_int("edgefactor", 16)) * n;
        const double pa = a.get_double("a", 0.57), pb = a.get_double("b", 0.19), pc = a.get_double("c", 0.19);
        SplitMix64 rng{uint64_t(a.get_int("seed", 1))};
        double t0 = wall_time();

        std::vector<uint32_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0u);
        for (uint64_t i = n - 1; i > 0; --i) std::swap(perm[i], perm[rng.next() % (i + 1)]);

        std::vector<uint64_t> keys(target);
        for (uint64_t e = 0; e < target; ++e) {
            uint64_t src = 0, dst = 0;
            for (int level = 0; level < scale; ++level) {
                double r = rng.uniform();
                uint64_t bit_src = r >= pa + pb;                    // quadrants c, d
                uint64_t bit_dst = (r >= pa && r < pa + pb) || r >= pa + pb + pc;  // quadrants b, d
                src = src << 1 | bit_src;
                dst = dst << 1 | bit_dst;
            }
            keys[e] = uint64_t(perm[dst]) << 32 | perm[src];
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

        Graph g = graph_from_keys(n, keys);
        write_graph(out, g);
        std::remove(ids_path(out).c_str());  // identity id map

        uint64_t dangling = 0;
        for (uint64_t v = 0; v < n; ++v) dangling += (g.out_deg[v] == 0);
        std::fprintf(stderr, "wrote %s: n=%" PRIu64 " m=%" PRIu64 " (generated %" PRIu64 ", dangling %" PRIu64 ") in %.1fs\n",
                     out.c_str(), n, g.m, target, dangling, wall_time() - t0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gen_rmat: %s\n", e.what());
        return 1;
    }
}
