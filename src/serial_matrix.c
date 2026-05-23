#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define MAKE_DIR(p) _mkdir(p)
static double wall_time(void) {
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
#include <sys/stat.h>
#define MAKE_DIR(p) mkdir(p, 0755)
static double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
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

    printf("Serial GEMM | N=%d\n", N);

    double t0 = wall_time();

    // i-k-j ordering: B accessed sequentially across j -> cache-friendly
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i*N + k];
            for (int j = 0; j < N; j++) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }

    double t1 = wall_time();
    double elapsed = t1 - t0;
    double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);
    printf("Time: %.6f s | Performance: %.3f GFLOPS\n", elapsed, gflops);

    (void)MAKE_DIR("outputs");
    FILE *fout = fopen("outputs/serial_output.txt", "w");
    if (!fout) { perror("File open failed"); free(A); free(B); free(C); return 1; }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) fprintf(fout, "%f ", C[i*N + j]);
        fprintf(fout, "\n");
    }
    fclose(fout);

    free(A); free(B); free(C);
    return 0;
}
