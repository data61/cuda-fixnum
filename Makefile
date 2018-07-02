INCLUDE_DIRS = -I./src
NVCC_FLAGS = -ccbin clang-3.8 -Wno-deprecated-declarations -std=c++11 -Xcompiler -Wall,-Wextra
NVCC_OPT_FLAGS = -lineinfo
NVCC_DBG_FLAGS = -g -G
NVCC_LIBS = -lstdc++ -lgtest
GENCODES = 50

% : %.cu
	nvcc $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -lgmp -o $@ $<

all:
	@echo "Please run 'make check' or 'make bench'."

tests/test-suite: tests/test-suite.cu
check: tests/test-suite
	@./tests/test-suite

bench/bench: bench/bench.cu
bench: bench/bench

.PHONY: clean
clean:
	$(RM) tests/test-suite bench/bench
