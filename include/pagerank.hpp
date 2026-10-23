// PageRank parameters and helpers shared by the sequential and MPI programs.
//
// Both programs compute, for every vertex v,
//
//     r'[v] = (1 - d) / N  +  d * ( sum_{u -> v} r[u] / outdeg[u]  +  D / N )
//
// where D is the total rank held by dangling vertices (outdeg = 0), which is
// spread uniformly over all vertices. Iteration stops when the L1 norm of
// (r' - r) drops below tol, or after max_iter iterations.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "graph.hpp"
#include "json.hpp"

namespace pr {

struct Params {
    double damping = 0.85;
    double tol = 1e-9;
    int max_iter = 200;
    int fixed_iters = 0;  // > 0: run exactly this many iterations (timing runs)
    int topk = 10;

    int iteration_limit() const { return fixed_iters > 0 ? fixed_iters : max_iter; }
    bool should_stop(double residual) const { return fixed_iters <= 0 && residual < tol; }
};

inline Params params_from_args(const Args& a) {
    Params p;
    p.damping = a.get_double("damping", p.damping);
    p.tol = a.get_double("tol", p.tol);
    p.max_iter = int(a.get_int("max-iter", p.max_iter));
    p.fixed_iters = int(a.get_int("iters", 0));
    p.topk = int(a.get_int("topk", p.topk));
    return p;
}

struct RankedNode {
    double score;
    uint64_t id;  // compact (0-based) vertex id
};

// Higher score first; ties broken by smaller id so results are deterministic.
inline bool ranks_before(const RankedNode& a, const RankedNode& b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

// Top-k of r[0..n) using a size-k heap, O(n log k). Ids are offset by `base`
// so a process can report global ids for its local slice.
inline std::vector<RankedNode> top_k(const double* r, uint64_t n, uint64_t base, int k) {
    std::vector<RankedNode> heap;
    heap.reserve(size_t(k) + 1);
    for (uint64_t i = 0; i < n; ++i) {
        RankedNode x{r[i], base + i};
        if (heap.size() < size_t(k)) {
            heap.push_back(x);
            std::push_heap(heap.begin(), heap.end(), ranks_before);
        } else if (k > 0 && ranks_before(x, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), ranks_before);
            heap.back() = x;
            std::push_heap(heap.begin(), heap.end(), ranks_before);
        }
    }
    std::sort(heap.begin(), heap.end(), ranks_before);
    return heap;
}

inline std::vector<RankedNode> merge_top_k(std::vector<RankedNode> candidates, int k) {
    std::sort(candidates.begin(), candidates.end(), ranks_before);
    if (candidates.size() > size_t(k)) candidates.resize(k);
    return candidates;
}

inline void print_top_k(const std::vector<RankedNode>& top, const IdMap& ids) {
    std::printf("\n  %-5s %-14s %-14s %s\n", "rank", "node (orig id)", "compact id", "PageRank");
    for (size_t i = 0; i < top.size(); ++i)
        std::printf("  %-5zu %-14llu %-14llu %.10f\n", i + 1, (unsigned long long)ids.original(top[i].id),
                    (unsigned long long)top[i].id, top[i].score);
}

inline std::string top_k_json(const std::vector<RankedNode>& top, const IdMap& ids) {
    std::vector<std::string> items;
    for (const auto& t : top) {
        JsonObject o;
        o.num("id", (unsigned long long)t.id)
            .num("orig", (unsigned long long)ids.original(t.id))
            .num("score", t.score);
        items.push_back(o.dump());
    }
    return json_list(items);
}

inline std::string basename_of(const std::string& path) {
    size_t s = path.find_last_of("/\\");
    std::string b = s == std::string::npos ? path : path.substr(s + 1);
    size_t dot = b.rfind(".bin");
    return dot == std::string::npos ? b : b.substr(0, dot);
}

}  // namespace pr
