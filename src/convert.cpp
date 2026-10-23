// convert: SNAP-style edge list -> binary CSR graph (see include/graph.hpp).
//
//   pigz -dc web-Google.txt.gz | ./build/convert web-Google.bin
//   ./build/convert out.bin --input edges.txt [--undirected]
//
// Input lines are "src dst" (any whitespace, extra columns ignored); lines
// starting with '#' or '%' are comments. Node ids are remapped to 0..n-1 in
// increasing order of original id, duplicate edges are removed and
// self-loops are kept.
#include <cinttypes>
#include <cstdio>
#include <vector>

#include "cli.hpp"
#include "graph.hpp"
#include "timer.hpp"

using namespace pr;

// Streams the edge list and appends (src, dst) pairs to `raw`.
static void parse_edges(FILE* in, std::vector<uint64_t>& raw) {
    std::vector<char> buf(1 << 24);
    uint64_t cur = 0, first = 0;
    bool in_num = false, comment = false;
    int tok = 0;
    auto end_number = [&] {
        if (tok == 0) first = cur;
        else if (tok == 1) { raw.push_back(first); raw.push_back(cur); }
        ++tok;
        cur = 0;
        in_num = false;
    };
    size_t got;
    while ((got = std::fread(buf.data(), 1, buf.size(), in)) > 0) {
        for (size_t i = 0; i < got; ++i) {
            char c = buf[i];
            if (comment) {
                if (c == '\n') { comment = false; tok = 0; }
                continue;
            }
            if (c >= '0' && c <= '9') {
                cur = cur * 10 + uint64_t(c - '0');
                in_num = true;
                continue;
            }
            if (in_num) end_number();
            if (c == '\n') tok = 0;
            else if ((c == '#' || c == '%') && tok == 0) comment = true;
        }
    }
    if (in_num) end_number();
}

int main(int argc, char** argv) {
    try {
        Args a(argc, argv);
        if (a.positional().size() != 1) {
            std::fprintf(stderr, "usage: convert <out.bin> [--input edges.txt] [--undirected]\n");
            return 2;
        }
        const std::string out = a.positional()[0];
        double t0 = wall_time();

        FILE* in = stdin;
        if (a.has("input")) {
            in = std::fopen(a.get("input").c_str(), "rb");
            if (!in) die("cannot open " + a.get("input"));
        }
        std::vector<uint64_t> raw;
        parse_edges(in, raw);
        if (in != stdin) std::fclose(in);
        const uint64_t input_edges = raw.size() / 2;
        std::fprintf(stderr, "parsed %" PRIu64 " edges in %.1fs\n", input_edges, wall_time() - t0);

        // Remap original ids to compact 0..n-1 ids, preserving id order.
        uint64_t max_id = 0;
        for (uint64_t x : raw) max_id = std::max(max_id, x);
        std::vector<uint64_t> orig;  // compact id -> original id
        if (max_id <= std::max<uint64_t>(4 * raw.size(), 1u << 20)) {
            std::vector<uint32_t> map(max_id + 1, UINT32_MAX);
            for (uint64_t x : raw) map[x] = 0;
            for (uint64_t x = 0; x <= max_id; ++x)
                if (map[x] == 0) { map[x] = uint32_t(orig.size()); orig.push_back(x); }
            for (uint64_t& x : raw) x = map[x];
        } else {  // sparse id space: sort-unique, then binary search
            orig = raw;
            std::sort(orig.begin(), orig.end());
            orig.erase(std::unique(orig.begin(), orig.end()), orig.end());
            for (uint64_t& x : raw) x = uint64_t(std::lower_bound(orig.begin(), orig.end(), x) - orig.begin());
        }
        const uint64_t n = orig.size();
        if (n >= UINT32_MAX) die("graph has too many vertices for 32-bit ids");

        // Pack each edge as (dst << 32 | src); sorting groups in-edges by destination.
        const bool undirected = a.has("undirected");
        std::vector<uint64_t> keys;
        keys.reserve(undirected ? raw.size() : raw.size() / 2);
        uint64_t self_loops = 0;
        for (size_t i = 0; i < raw.size(); i += 2) {
            uint64_t s = raw[i], d = raw[i + 1];
            self_loops += (s == d);
            keys.push_back(d << 32 | s);
            if (undirected && s != d) keys.push_back(s << 32 | d);
        }
        std::vector<uint64_t>().swap(raw);
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

        Graph g = graph_from_keys(n, keys);
        write_graph(out, g);
        write_ids(ids_path(out), orig);

        uint64_t dangling = 0, max_in = 0, max_out = 0;
        for (uint64_t v = 0; v < n; ++v) {
            dangling += (g.out_deg[v] == 0);
            max_in = std::max(max_in, g.in_off[v + 1] - g.in_off[v]);
            max_out = std::max<uint64_t>(max_out, g.out_deg[v]);
        }
        std::fprintf(stderr,
                     "wrote %s: n=%" PRIu64 " m=%" PRIu64 " (input %" PRIu64 ", self-loops %" PRIu64
                     ", dangling %" PRIu64 ", max in-deg %" PRIu64 ", max out-deg %" PRIu64 ") in %.1fs\n",
                     out.c_str(), n, g.m, input_edges, self_loops, dangling, max_in, max_out, wall_time() - t0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "convert: %s\n", e.what());
        return 1;
    }
}
