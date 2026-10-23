// Wall-clock timer for the non-MPI tools (the MPI program uses MPI_Wtime).
#pragma once

#include <chrono>

namespace pr {

inline double wall_time() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace pr
