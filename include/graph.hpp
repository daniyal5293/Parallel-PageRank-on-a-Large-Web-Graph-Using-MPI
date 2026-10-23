// Binary CSR graph format shared by every tool in the project.
//
// On-disk layout (little-endian):
//   Header   { char magic[8] = "PRCSR1"; u64 n; u64 m; }
//   u64 in_off[n+1]   CSR offsets of in-edges, grouped by destination vertex
//   u32 in_src[m]     source vertex of every in-edge
//   u32 out_deg[n]    out-degree of every vertex
//
// The layout is "pull" oriented: vertex v's new rank is computed from the
// sources listed in in_src[in_off[v] .. in_off[v+1]), so each vertex is only
// ever written by one process and no locking or atomics are needed.
//
// An optional side file "<graph>.ids" holds u64 orig_id[n] so results can be
// reported with the original SNAP node ids instead of compact ids.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pr {

struct Header {
    char magic[8];
    uint64_t n;
    uint64_t m;
};
static_assert(sizeof(Header) == 24, "unexpected header padding");

inline constexpr char kMagic[8] = {'P', 'R', 'C', 'S', 'R', '1', 0, 0};

// Byte positions of each array inside the file.
inline uint64_t in_off_pos(const Header&) { return sizeof(Header); }
inline uint64_t in_src_pos(const Header& h) { return sizeof(Header) + 8 * (h.n + 1); }
inline uint64_t out_deg_pos(const Header& h) { return in_src_pos(h) + 4 * h.m; }

struct Graph {
    uint64_t n = 0, m = 0;
    std::vector<uint64_t> in_off;
    std::vector<uint32_t> in_src;
    std::vector<uint32_t> out_deg;
};

[[noreturn]] inline void die(const std::string& msg) { throw std::runtime_error(msg); }

inline void check_header(const Header& h, const std::string& path) {
    if (std::memcmp(h.magic, kMagic, 6) != 0) die(path + ": not a PRCSR1 graph file");
    if (h.n >= UINT32_MAX) die(path + ": too many vertices for 32-bit ids");
}

inline std::string ids_path(const std::string& graph_path) { return graph_path + ".ids"; }

template <class T>
void fwrite_all(FILE* f, const T* p, size_t count, const std::string& path) {
    if (count && std::fwrite(p, sizeof(T), count, f) != count) die(path + ": write failed");
}

template <class T>
void fread_all(FILE* f, T* p, size_t count, const std::string& path) {
    if (count && std::fread(p, sizeof(T), count, f) != count) die(path + ": truncated file");
}

inline Graph read_graph(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open " + path);
    Header h;
    fread_all(f, &h, 1, path);
    check_header(h, path);
    Graph g;
    g.n = h.n;
    g.m = h.m;
    g.in_off.resize(h.n + 1);
    g.in_src.resize(h.m);
    g.out_deg.resize(h.n);
    fread_all(f, g.in_off.data(), g.in_off.size(), path);
    fread_all(f, g.in_src.data(), g.in_src.size(), path);
    fread_all(f, g.out_deg.data(), g.out_deg.size(), path);
    std::fclose(f);
    return g;
}

inline void write_graph(const std::string& path, const Graph& g) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) die("cannot create " + path);
    Header h{};
    std::memcpy(h.magic, kMagic, 8);
    h.n = g.n;
    h.m = g.m;
    fwrite_all(f, &h, 1, path);
    fwrite_all(f, g.in_off.data(), g.in_off.size(), path);
    fwrite_all(f, g.in_src.data(), g.in_src.size(), path);
    fwrite_all(f, g.out_deg.data(), g.out_deg.size(), path);
    if (std::fclose(f) != 0) die(path + ": write failed");
}

// Builds the CSR from edge keys packed as (dst << 32 | src). The keys must be
// sorted and free of duplicates, which also leaves each in-edge list sorted.
inline Graph graph_from_keys(uint64_t n, const std::vector<uint64_t>& keys) {
    Graph g;
    g.n = n;
    g.m = keys.size();
    g.in_off.assign(n + 1, 0);
    g.in_src.resize(keys.size());
    g.out_deg.assign(n, 0);
    for (size_t e = 0; e < keys.size(); ++e) {
        uint32_t dst = uint32_t(keys[e] >> 32), src = uint32_t(keys[e]);
        g.in_off[dst + 1]++;
        g.in_src[e] = src;
        g.out_deg[src]++;
    }
    for (uint64_t v = 0; v < n; ++v) g.in_off[v + 1] += g.in_off[v];
    return g;
}

inline void write_ids(const std::string& path, const std::vector<uint64_t>& ids) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) die("cannot create " + path);
    fwrite_all(f, ids.data(), ids.size(), path);
    if (std::fclose(f) != 0) die(path + ": write failed");
}

// Reads the whole id map of a graph (identity when there is no .ids file).
inline std::vector<uint64_t> read_ids(const std::string& graph_path, uint64_t n) {
    std::vector<uint64_t> ids(n);
    FILE* f = std::fopen(ids_path(graph_path).c_str(), "rb");
    if (!f) {
        for (uint64_t i = 0; i < n; ++i) ids[i] = i;
        return ids;
    }
    fread_all(f, ids.data(), n, ids_path(graph_path));
    std::fclose(f);
    return ids;
}

// Maps compact ids back to original ids. Falls back to identity when the
// graph has no .ids file (e.g. synthetic R-MAT graphs).
class IdMap {
public:
    explicit IdMap(const std::string& graph_path) : f_(std::fopen(ids_path(graph_path).c_str(), "rb")) {}
    ~IdMap() { if (f_) std::fclose(f_); }
    IdMap(const IdMap&) = delete;
    IdMap& operator=(const IdMap&) = delete;

    uint64_t original(uint64_t v) const {
        if (!f_) return v;
        uint64_t id = v;
        if (std::fseek(f_, long(v * sizeof(uint64_t)), SEEK_SET) == 0 &&
            std::fread(&id, sizeof(id), 1, f_) == 1)
            return id;
        return v;
    }

private:
    FILE* f_;
};

}  // namespace pr
