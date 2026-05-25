#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
  #include <direct.h>
  #define MAKE_DIR(p) _mkdir(p)
#else
  #include <sys/stat.h>
  #define MAKE_DIR(p) mkdir(p, 0755)
#endif

int main(int argc, char *argv[])
{
    int rank, size;

    // Start MPI and find this process ID (rank) and total process count (size).
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Matrix size comes from the command line; default is 1024 if not given.
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid N=%d\n", N);
        MPI_Finalize();
        return 1;
    }

    // Scatterv/Gatherv counts & displacements (handles N % size != 0)
    int *row_counts  = (int*)malloc(size * sizeof(int));
    int *elem_counts = (int*)malloc(size * sizeof(int));
    int *elem_displs = (int*)malloc(size * sizeof(int));
    int base = N / size;
    int rem  = N % size;
    int offset_rows = 0;
    for (int r = 0; r < size; r++) {
        // Extra rows are given to the first "rem" ranks when N is not divisible.
        row_counts[r]  = base + (r < rem ? 1 : 0);
        elem_counts[r] = row_counts[r] * N;
        elem_displs[r] = offset_rows * N;
        offset_rows   += row_counts[r];
    }
    int my_rows = row_counts[rank];

    // B is needed by every rank. A and C exist fully only on rank 0.
    // Each rank stores only its assigned rows of A and C in local_A/local_C.
    double *A = NULL, *C = NULL;
    double *B       = (double*)malloc((size_t)N * N * sizeof(double));
    double *local_A = (double*)malloc((size_t)my_rows * N * sizeof(double));
    double *local_C = (double*)calloc((size_t)my_rows * N, sizeof(double));

    if (rank == 0) {
        // Rank 0 creates the full input matrices and later receives the full result.
        A = (double*)malloc((size_t)N * N * sizeof(double));
        C = (double*)malloc((size_t)N * N * sizeof(double));
        srand(1);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = rand() % 10;
                B[i*N + j] = rand() % 10;
            }
        }
        printf("MPI GEMM | N=%d | processes=%d\n", N, size);
    }

    // Synchronize all ranks before timing the parallel section.
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    // Send the full B matrix from rank 0 to every process.
    // Every rank needs B to compute its own rows of C.
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_bcast = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    // Split matrix A by rows and send each rank only the rows it must compute.
    // Scatterv is used because row counts may be uneven.
    MPI_Scatterv(A, elem_counts, elem_displs, MPI_DOUBLE,
                 local_A, my_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    double t_scatter = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    // Local matrix multiplication:
    // each rank computes local_C = local_A * B for its assigned rows.
    for (int i = 0; i < my_rows; i++) {
        for (int k = 0; k < N; k++) {
            double a = local_A[i*N + k];
            for (int j = 0; j < N; j++) {
                local_C[i*N + j] += a * B[k*N + j];
            }
        }
    }
    double t_compute = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    // Collect all computed row blocks back into the final C matrix on rank 0.
    MPI_Gatherv(local_C, my_rows * N, MPI_DOUBLE,
                C, elem_counts, elem_displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    double t_gather = MPI_Wtime() - t0;

    double t_total = MPI_Wtime() - t_start;

    // Aggregate phase times (max across ranks => worst-case wall time per phase)
    double mx_bcast, mx_scatter, mx_compute, mx_gather;
    MPI_Reduce(&t_bcast,   &mx_bcast,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_scatter, &mx_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compute, &mx_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_gather,  &mx_gather,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Matrix multiplication performs about 2*N^3 floating-point operations.
        double gflops = (2.0 * (double)N * (double)N * (double)N) / (t_total * 1e9);
        double phase_sum = mx_bcast + mx_scatter + mx_compute + mx_gather;
        printf("Total: %.6f s | Performance: %.3f GFLOPS\n", t_total, gflops);
        printf("Phase breakdown (per-phase max across ranks; %% of phase-sum):\n");
        printf("  Broadcast B : %.6f s (%5.1f%%)\n", mx_bcast,   100.0 * mx_bcast   / phase_sum);
        printf("  Scatter   A : %.6f s (%5.1f%%)\n", mx_scatter, 100.0 * mx_scatter / phase_sum);
        printf("  Compute     : %.6f s (%5.1f%%)\n", mx_compute, 100.0 * mx_compute / phase_sum);
        printf("  Gather    C : %.6f s (%5.1f%%)\n", mx_gather,  100.0 * mx_gather  / phase_sum);
        printf("  Comm total  : %.6f s (%5.1f%%)\n",
               mx_bcast + mx_scatter + mx_gather,
               100.0 * (mx_bcast + mx_scatter + mx_gather) / phase_sum);

        (void)MAKE_DIR("outputs");
        // Save final matrix so it can be compared with the serial result.
        FILE *fout = fopen("outputs/mpi_output.txt", "w");
        if (fout) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) fprintf(fout, "%f ", C[i*N + j]);
                fprintf(fout, "\n");
            }
            fclose(fout);
        }
    }

    free(B); free(local_A); free(local_C);
    free(row_counts); free(elem_counts); free(elem_displs);
    if (rank == 0) { free(A); free(C); }
    // Shut down MPI before exiting.
    MPI_Finalize();
    return 0;
}
