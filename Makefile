# Makefile for the GAPBS direction-optimizing BFS ported to Caladan.
# Build from the Caladan tree root first (`make` to produce libbase.a,
# libruntime.a, libnet.a and bindings/cc/librt++.a), then `make` here.
ROOT_PATH=../..
include $(ROOT_PATH)/build/shared.mk

# C++ runtime bindings (rt::Thread, rt::WaitGroup, rt::RuntimeInit, ...).
librt_libs = $(ROOT_PATH)/bindings/cc/librt++.a
INC += -I$(ROOT_PATH)/bindings/cc

# The vendored GAPBS support headers live alongside this file.
INC += -I$(CURDIR)/gapbs

# GAPBS selects real atomic compare_and_swap / fetch_and_add in
# platform_atomics.h when _OPENMP is defined.  We define it WITHOUT -fopenmp
# so we get the atomics only -- the parallelism comes from Caladan threads, not
# libgomp, and the (now ignored) "#pragma omp" hints in the GAPBS headers fall
# back to serial graph construction, which is fine (build time is untimed).
CXXFLAGS += -D_OPENMP

bfs_src = bfs.cc
bfs_obj = $(bfs_src:.cc=.o)

# Build the GAPBS headers with C++ (they are C++-only) using the same flags.
$(bfs_obj): $(bfs_src) $(wildcard gapbs/*.h)
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: bfs

bfs: $(bfs_obj) $(librt_libs) $(RUNTIME_DEPS)
	$(LDXX) -o $@ $(LDFLAGS) $(bfs_obj) $(librt_libs) $(RUNTIME_LIBS)

clean:
	rm -f bfs $(bfs_obj)

.PHONY: all clean
