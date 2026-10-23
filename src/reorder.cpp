// reorder: relabels the vertices of a binary CSR graph for cache locality.
//
//   ./build/reorder in.bin out.bin [--order degree]
//
// In the pull formulation, contrib[u] is read once per out-edge of u, i.e.
// out_deg[u] times per iteration. Numbering vertices by decreasing out-degree
// packs the most frequently read values into a small, cache-resident prefix
// of the contribution array. The .ids side file is composed so results are
// still reported with the original node ids.
#include <cinttypes>
#include <cstdio>
#include <numeric>
#include <vector>

#include "cli.hpp"
#include "graph.hpp"
#include "timer.hpp"

using namespace pr;

int main(int argc, char** argv) {
    try {
        Args a(argc, argv);
        if (a.positional().size() != 2) {
            std::fprintf(stderr, "usage: reorder <in.bin> <out.bin> [--order degree]\n");
            return 2;
        }
        const std::string in = a.positional()[0], out = a.positional()[1];
        if (a.get("order", "degree") != "degree") die("only --order degree is supported");
        double t0 = wall_time();
        Graph g = read_graph(in);
        const uint64_t n = g.n;

        std::vector<uint32_t> by_rank(n);  // new id -> old id
        std::iota(by_rank.begin(), by_rank.end(), 0u);
        std::stable_sort(by_rank.begin(), by_rank.end(),
                         [&](uint32_t x, uint32_t y) { return g.out_deg[x] > g.out_deg[y]; });
        std::vector<uint32_t> new_id(n);  // old id -> new id
        for (uint64_t i = 0; i < n; ++i) new_id[by_rank[i]] = uint32_t(i);

        std::vector<uint64_t> keys;
        keys.reserve(g.m);
        for (uint64_t v = 0; v < n; ++v)
            for (uint64_t e = g.in_off[v]; e < g.in_off[v + 1]; ++e)
                keys.push_back(uint64_t(new_id[v]) << 32 | new_id[g.in_src[e]]);
        std::vector<uint32_t>().swap(g.in_src);
        std::sort(keys.begin(), keys.end());
        write_graph(out, graph_from_keys(n, keys));

        const std::vector<uint64_t> old_ids = read_ids(in, n);
        std::vector<uint64_t> ids(n);
        for (uint64_t i = 0; i < n; ++i) ids[i] = old_ids[by_rank[i]];
        write_ids(ids_path(out), ids);
        std::fprintf(stderr, "wrote %s: n=%" PRIu64 " m=%" PRIu64 " (degree order) in %.1fs\n", out.c_str(), n,
                     uint64_t(keys.size()), wall_time() - t0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "reorder: %s\n", e.what());
        return 1;
    }
}
