# Makefile for NVIB Cloaking Library

CC = gcc
CFLAGS = -O3 -ffast-math -fPIC -Wall -Wextra -std=c11
LDFLAGS = -lm
LIB_DIR = infemeral/nvib

# Detect SIMD support at compile time
HAS_AVX512 := $(shell grep -q avx512 /proc/cpuinfo 2>/dev/null && echo yes || echo no)
HAS_AVX2 := $(shell grep -q avx2 /proc/cpuinfo 2>/dev/null && echo yes || echo no)

ifeq ($(HAS_AVX512),yes)
	SIMD_DEFINES = -D__AVX512F__
	SIMD_FLAGS = -mavx512f -mavx512cd -mavx2 -msse4.1
else ifeq ($(HAS_AVX2),yes)
	SIMD_DEFINES = -D__AVX2__
	SIMD_FLAGS = -mavx2 -msse4.1
else
	SIMD_DEFINES = -D__SSE4_1__
	SIMD_FLAGS = -msse4.1
endif

# Target: Build shared library
$(LIB_DIR)/nvib_cloak.so: $(LIB_DIR)/nvib_cloak.c $(LIB_DIR)/nvib_cloak.h
	@echo "Building NVIB cloaking library..."
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $(SIMD_DEFINES) -shared -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

# Target: Build everything
all: $(LIB_DIR)/nvib_cloak.so

# Target: Clean build artifacts
clean:
	rm -f $(LIB_DIR)/nvib_cloak.so
	@echo "Cleaned build artifacts"

# Target: Check if library exists
check:
	@if [ -f $(LIB_DIR)/nvib_cloak.so ]; then \
		echo "Library exists: $(LIB_DIR)/nvib_cloak.so"; \
	else \
		echo "Library not found. Run 'make' to build."; \
		exit 1; \
	fi

.PHONY: all clean check
