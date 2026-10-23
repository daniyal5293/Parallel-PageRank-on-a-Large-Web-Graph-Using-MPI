# Parallel PageRank (MPI) - build file.
#   make            build everything into build/
#   make test       build, then run the correctness test suite
#   make clean      remove build/
CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
BUILD    := build
HEADERS  := $(wildcard include/*.hpp)

PROGRAMS := $(BUILD)/pagerank_seq $(BUILD)/pagerank_mpi $(BUILD)/convert $(BUILD)/gen_rmat $(BUILD)/reorder \
            $(BUILD)/platform_probe

.PHONY: all test clean
all: $(PROGRAMS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/platform_probe: src/platform_probe.cpp $(HEADERS) | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -Iinclude $< -o $@

$(BUILD)/pagerank_mpi: src/pagerank_mpi.cpp $(HEADERS) | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -Iinclude $< -o $@

$(BUILD)/%: src/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@

test: all
	bash tests/run_tests.sh

clean:
	rm -rf $(BUILD)
