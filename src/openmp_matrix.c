#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifdef _WIN32
  #include <direct.h>
  #define MAKE_DIR(p) _mkdir(p)
#else
  #include <sys/stat.h>
  #define MAKE_DIR(p) mkdir(p, 0755)
#endif

int main(int argc, char *argv[])
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N <= 0) { fprintf(stderr, "Invalid N=%d\n", N); return 1; }

    double *A = (double*)malloc((size_t)N * N * sizeof(double));
    double *B = (double*)malloc((size_t)N * N * sizeof(double));
    double *C = (double*)calloc((size_t)N * N, sizeof(double));
    if (!A || !B || !C) { fprintf(stderr, "Allocation failed for N=%d\n", N); return 1; }

    srand(1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = rand() % 10;
            B[i*N + j] = rand() % 10;
        }
    }

    int nthreads = omp_get_max_threads();
    printf("OpenMP GEMM | N=%d | threads=%d\n", N, nthreads);

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i*N + k];
            for (int j = 0; j < N; j++) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }

    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;
    double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);
    printf("Time: %.6f s | Performance: %.3f GFLOPS\n", elapsed, gflops);

    (void)MAKE_DIR("outputs");
    FILE *fout = fopen("outputs/openmp_output.txt", "w");
    if (!fout) { perror("File open failed"); free(A); free(B); free(C); return 1; }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) fprintf(fout, "%f ", C[i*N + j]);
        fprintf(fout, "\n");
    }
    fclose(fout);

    free(A); free(B); free(C);
    return 0;
}
