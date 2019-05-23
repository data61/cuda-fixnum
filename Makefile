CXX ?= g++
INCLUDE_DIRS = -I./src
NVCC_FLAGS = -ccbin $(CXX) -std=c++11 -Xcompiler -Wall,-Wextra
NVCC_OPT_FLAGS = -DNDEBUG
NVCC_TEST_FLAGS = -lineinfo
NVCC_DBG_FLAGS = -g -G
NVCC_LIBS = -lstdc++
NVCC_TEST_LIBS = -lgtest

# SOURCE: https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html#building-turing-compatible-apps-using-cuda-10-0
# NOTE: Supports 6 different chipsets, but results in binary sizes
#   7x larger (21MB:3MB) and build times 7X slower (80s:12s).
# DECODE:
# 50==Maxwell:Tesla/Quadro, 52==Maxewll,GTX 9**, 60==Pascal:Tesla P100
# 61==Pascal:GTX 10**, 70==Volta:Tesla V100, 75==Turing:RTX 20**

NVCC_TURING_COMPAT_MODE = -arch=sm_50 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_75,code=compute_75


# Check that nvcc is in path
ifeq (, $(shell which nvcc))
 $(error "No nvcc in $$PATH, consider doing: export PATH=$$PATH:/usr/local/cuda/bin")
endif


all:
	@echo "Please run 'make check' or 'make bench'."

tests/test-suite: tests/test-suite.cu
	nvcc $(NVCC_TEST_FLAGS) \
		$(NVCC_FLAGS) \
		$(NVCC_TURING_COMPAT_MODE) \
		$(INCLUDE_DIRS) \
		$(NVCC_LIBS) \
		$(NVCC_TEST_LIBS) \
		-o $@ $<

check: tests/test-suite
	@./tests/test-suite

bench/bench: bench/bench.cu
	nvcc $(NVCC_OPT_FLAGS) \
		$(NVCC_FLAGS) \
		$(NVCC_TURING_COMPAT_MODE) \
		$(INCLUDE_DIRS) \
		$(NVCC_LIBS) \
		-o $@ $<

bench: bench/bench

.PHONY: clean
clean:
	$(RM) tests/test-suite bench/bench
