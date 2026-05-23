#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

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

typedef struct {
    int row_start;
    int row_end;
    int N;
    const double *A;
    const double *B;
    double *C;
} thread_arg_t;

static void* worker(void *arg)
{
    thread_arg_t *t = (thread_arg_t*)arg;
    int N = t->N;
    for (int i = t->row_start; i < t->row_end; i++) {
        for (int k = 0; k < N; k++) {
            double a = t->A[i*N + k];
            for (int j = 0; j < N; j++) {
                t->C[i*N + j] += a * t->B[k*N + j];
            }
        }
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    int N        = (argc > 1) ? atoi(argv[1]) : 1024;
    int nthreads = (argc > 2) ? atoi(argv[2]) : 4;
    if (N <= 0 || nthreads <= 0) {
        fprintf(stderr, "Usage: %s [N=1024] [threads=4]\n", argv[0]);
        return 1;
    }

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

    printf("Pthreads GEMM | N=%d | threads=%d\n", N, nthreads);

    pthread_t    *tids = (pthread_t*)   malloc(nthreads * sizeof(pthread_t));
    thread_arg_t *args = (thread_arg_t*)malloc(nthreads * sizeof(thread_arg_t));

    int base = N / nthreads;
    int rem  = N % nthreads;
    int offset = 0;

    double t0 = wall_time();

    for (int t = 0; t < nthreads; t++) {
        int rows = base + (t < rem ? 1 : 0);
        args[t].row_start = offset;
        args[t].row_end   = offset + rows;
        args[t].N = N;
        args[t].A = A; args[t].B = B; args[t].C = C;
        offset += rows;
        pthread_create(&tids[t], NULL, worker, &args[t]);
    }
    for (int t = 0; t < nthreads; t++) {
        pthread_join(tids[t], NULL);
    }

    double t1 = wall_time();
    double elapsed = t1 - t0;
    double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);
    printf("Time: %.6f s | Performance: %.3f GFLOPS\n", elapsed, gflops);

    (void)MAKE_DIR("outputs");
    FILE *fout = fopen("outputs/pthreads_output.txt", "w");
    if (!fout) { perror("File open failed"); free(A); free(B); free(C); free(tids); free(args); return 1; }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) fprintf(fout, "%f ", C[i*N + j]);
        fprintf(fout, "\n");
    }
    fclose(fout);

    free(tids); free(args);
    free(A); free(B); free(C);
    return 0;
}
