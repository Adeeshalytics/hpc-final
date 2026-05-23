# Makefile for HPC matrix multiplication project (MS-MPI + MinGW gcc on Windows)
#
# Project layout:
#   src/     - C source files
#   bin/     - compiled executables (auto-created)
#   scripts/ - Python tooling
#   results/ - benchmark CSV outputs
#   outputs/ - per-variant matrix dumps for RMSE check (auto-created)
#   figures/ - charts and architecture diagrams
#
# Build:
#   make all                     - build every variant
#   make serial / openmp / ...   - build a single variant
#
# Run:
#   make run-serial    N=1024
#   make run-openmp    N=1024 THREADS=8
#   make run-pthreads  N=1024 THREADS=8
#   make run-mpi       N=1024 NP=4
#   make run-hybrid    N=1024 NP=2 THREADS=4
#
# Verify accuracy of all variants against serial:
#   make verify        N=1024 NP=4 THREADS=4
#
# Sweep problem size and configurations for the report:
#   make benchmark
#
# Clean:
#   make clean

# Force MSYS2 sh as the recipe shell so `VAR=value cmd` syntax works
# even when `mingw32-make` is invoked from PowerShell or cmd.exe.
SHELL       := C:/msys64/usr/bin/sh.exe
.SHELLFLAGS := -c

CC          = gcc
CFLAGS      = -O2 -Wall -Wextra
LDFLAGS     =

# MS-MPI install paths (adjust if yours differ)
MSMPI_ROOT  = C:/MPI/SDK
MPIEXEC     = C:/MPI/Bin/mpiexec.exe
MPI_CFLAGS  = -I"$(MSMPI_ROOT)/Include"
MPI_LDFLAGS = -L"$(MSMPI_ROOT)/Lib/x64" -lmsmpi

# Defaults (override on command line: make run-mpi N=2048 NP=8)
N       ?= 1024
NP      ?= 4
THREADS ?= 4

# Directory layout
SRC_DIR  = src
BIN_DIR  = bin
OUT_DIR  = outputs

EXES = $(BIN_DIR)/serial_gemm.exe \
       $(BIN_DIR)/openmp_matrix.exe \
       $(BIN_DIR)/pthreads_matrix.exe \
       $(BIN_DIR)/mpi_matrix.exe \
       $(BIN_DIR)/hybrid_matrix.exe

all: $(EXES)

# ---- Aliases for individual targets -----------------------------------------

serial:    $(BIN_DIR)/serial_gemm.exe
openmp:    $(BIN_DIR)/openmp_matrix.exe
pthreads:  $(BIN_DIR)/pthreads_matrix.exe
mpi:       $(BIN_DIR)/mpi_matrix.exe
hybrid:    $(BIN_DIR)/hybrid_matrix.exe

# ---- Order-only directory creation ------------------------------------------

$(BIN_DIR) $(OUT_DIR):
	@mkdir -p $@

# ---- Build rules ------------------------------------------------------------

$(BIN_DIR)/serial_gemm.exe: $(SRC_DIR)/serial_matrix.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/openmp_matrix.exe: $(SRC_DIR)/openmp_matrix.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

$(BIN_DIR)/pthreads_matrix.exe: $(SRC_DIR)/pthreads_matrix.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< -lpthread $(LDFLAGS)

$(BIN_DIR)/mpi_matrix.exe: $(SRC_DIR)/mpi_matrix.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(MPI_CFLAGS) -o $@ $< $(MPI_LDFLAGS) $(LDFLAGS)

$(BIN_DIR)/hybrid_matrix.exe: $(SRC_DIR)/hybrid_matrix.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -fopenmp $(MPI_CFLAGS) -o $@ $< $(MPI_LDFLAGS) $(LDFLAGS)

# ---- Run targets ------------------------------------------------------------

run-serial: $(BIN_DIR)/serial_gemm.exe | $(OUT_DIR)
	./$(BIN_DIR)/serial_gemm.exe $(N)

run-openmp: $(BIN_DIR)/openmp_matrix.exe | $(OUT_DIR)
	OMP_NUM_THREADS=$(THREADS) ./$(BIN_DIR)/openmp_matrix.exe $(N)

run-pthreads: $(BIN_DIR)/pthreads_matrix.exe | $(OUT_DIR)
	./$(BIN_DIR)/pthreads_matrix.exe $(N) $(THREADS)

run-mpi: $(BIN_DIR)/mpi_matrix.exe | $(OUT_DIR)
	"$(MPIEXEC)" -n $(NP) ./$(BIN_DIR)/mpi_matrix.exe $(N)

run-hybrid: $(BIN_DIR)/hybrid_matrix.exe | $(OUT_DIR)
	OMP_NUM_THREADS=$(THREADS) "$(MPIEXEC)" -n $(NP) ./$(BIN_DIR)/hybrid_matrix.exe $(N)

# ---- Verification: run all variants, then RMSE-check against serial ---------

verify: $(EXES) | $(OUT_DIR)
	./$(BIN_DIR)/serial_gemm.exe $(N)
	OMP_NUM_THREADS=$(THREADS) ./$(BIN_DIR)/openmp_matrix.exe $(N)
	./$(BIN_DIR)/pthreads_matrix.exe $(N) $(THREADS)
	"$(MPIEXEC)" -n $(NP) ./$(BIN_DIR)/mpi_matrix.exe $(N)
	OMP_NUM_THREADS=$(THREADS) "$(MPIEXEC)" -n $(NP) ./$(BIN_DIR)/hybrid_matrix.exe $(N)
	python scripts/rmse_check.py

# ---- Benchmark (full sweep + analysis) --------------------------------------

benchmark: $(EXES) | $(OUT_DIR)
	python scripts/run_benchmarks.py
	python scripts/analyze_benchmarks.py

# ---- Cleanup ----------------------------------------------------------------

clean:
	-rm -rf $(BIN_DIR) $(OUT_DIR)

.PHONY: all serial openmp pthreads mpi hybrid \
        run-serial run-openmp run-pthreads run-mpi run-hybrid \
        verify benchmark clean
