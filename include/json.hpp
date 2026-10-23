// Tiny JSON object writer used to hand run results to the benchmark scripts.
#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace pr {

class JsonObject {
public:
    JsonObject() { os_.precision(12); }

    JsonObject& num(const std::string& k, double v) { key(k); os_ << v; return *this; }
    JsonObject& num(const std::string& k, long long v) { key(k); os_ << v; return *this; }
    JsonObject& num(const std::string& k, unsigned long long v) { key(k); os_ << v; return *this; }
    JsonObject& num(const std::string& k, int v) { return num(k, (long long)v); }
    JsonObject& boolean(const std::string& k, bool v) { key(k); os_ << (v ? "true" : "false"); return *this; }
    JsonObject& str(const std::string& k, const std::string& v) { key(k); os_ << quote(v); return *this; }
    JsonObject& raw(const std::string& k, const std::string& json) { key(k); os_ << json; return *this; }

    template <class T>
    JsonObject& arr(const std::string& k, const std::vector<T>& v) {
        key(k);
        os_ << '[';
        for (size_t i = 0; i < v.size(); ++i) os_ << (i ? "," : "") << v[i];
        os_ << ']';
        return *this;
    }

    std::string dump() const { return "{" + os_.str() + "}"; }

    static std::string quote(const std::string& s) {
        std::string out = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') out += '\\';
            out += c;
        }
        return out + "\"";
    }

private:
    void key(const std::string& k) {
        if (!first_) os_ << ',';
        first_ = false;
        os_ << quote(k) << ':';
    }

    std::ostringstream os_;
    bool first_ = true;
};

inline std::string json_list(const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? "," : "") + items[i];
    return out + "]";
}

inline void write_text(const std::string& path, const std::string& text) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fputs(text.c_str(), f);
    std::fputc('\n', f);
    std::fclose(f);
}

}  // namespace pr
