// Minimal "--key value" / "--flag" command-line parser.
#pragma once

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace pr {

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                std::string key = a.substr(2);
                if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
                    kv_[key] = argv[++i];
                else
                    kv_[key] = "1";
            } else {
                positional_.push_back(a);
            }
        }
    }

    bool has(const std::string& k) const { return kv_.count(k) != 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    long long get_int(const std::string& k, long long def) const {
        return has(k) ? std::atoll(get(k).c_str()) : def;
    }
    double get_double(const std::string& k, double def) const {
        return has(k) ? std::atof(get(k).c_str()) : def;
    }
    const std::vector<std::string>& positional() const { return positional_; }

private:
    std::map<std::string, std::string> kv_;
    std::vector<std::string> positional_;
};

}  // namespace pr
