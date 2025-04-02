#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h> // Include windows.h for timing

void fwht_sequency(double* x, size_t N) {
    // First stage calculation
    for (size_t i = 0; i < N - 1; i += 2) {
        double temp1 = x[i];
        double temp2 = x[i + 1];
        x[i] = temp1 + temp2;
        x[i + 1] = temp1 - 2 * temp2;
    }

    // Subsequent stages
    size_t L = 1;
    double* y = (double*)malloc(N * sizeof(double));
    if (y == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return;
    }

    for (size_t nStage = 2; nStage <= log2(N); nStage++) {
        size_t M = 1 << L;
        size_t J = 0;
        size_t K = 0;

        while (K < N) {
            for (size_t j = J; j < J + M - 1; j += 2) {
                y[K] = x[j] + x[j + M];
                y[K + 1] = x[j] - x[j + M];
                y[K + 2] = x[j + 1] - x[j + 1 + M];
                y[K + 3] = x[j + 1] + x[j + 1 + M];
                K += 4;
            }
            J += 2 * M;
        }

        // Copy y back to x for the next stage
        memcpy(x, y, N * sizeof(double));
        L++;
    }

    free(y);

    // Normalization
    for (size_t i = 0; i < N; i++) {
        x[i] /= (double)N;
    }
}

void ifwht_sequency(double* x, size_t N) {
    fwht_sequency(x, N); // Forward transform is its own inverse (except for scaling)

    // Scale back up by N for inverse transform
    for (size_t i = 0; i < N; i++) {
        x[i] *= (double)N;
    }
}

void write_fwht_to_file(double* x, size_t N, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }

    for (size_t i = 0; i < N; i++) {
        fprintf(fp, "%.10f\n", x[i]);
    }

    fclose(fp);
    printf("FWHT results written to %s\n", filename);
}

int main() {
    LARGE_INTEGER StartingTime, EndingTime, Frequency;
    QueryPerformanceFrequency(&Frequency);
    double total_time, ave_time;
    const size_t ARRAY_SIZE = 1 << 10;
    const size_t loope = 5;

    double* x = (double*)malloc(ARRAY_SIZE * sizeof(double));
    if (x == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Initialize data
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        x[i] = (double)i;
    }

    printf("*** FWHT ***\n");
    printf("numElements = %lu\n", ARRAY_SIZE);

    // Store original data for file output
    double* fwht_results = (double*)malloc(ARRAY_SIZE * sizeof(double));
    if (fwht_results == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        free(x);
        return 1;
    }
    
    // Copy initial data
    memcpy(fwht_results, x, ARRAY_SIZE * sizeof(double));
    
    // Perform FWHT on the copy for file output
    fwht_sequency(fwht_results, ARRAY_SIZE);
    
    // Write results to file
    write_fwht_to_file(fwht_results, ARRAY_SIZE, "fwht_results.txt");
    
    // Free the results memory
    free(fwht_results);

    // Continue with benchmarking
    total_time = 0.0;
    for (size_t i = 0; i < loope; i++) {
        QueryPerformanceCounter(&StartingTime);
        fwht_sequency(x, ARRAY_SIZE);
        QueryPerformanceCounter(&EndingTime);
        total_time += ((double)((EndingTime.QuadPart - StartingTime.QuadPart) * 1000000 / Frequency.QuadPart)) / 1000;
        // reset the array before next run.
        for(size_t j = 0; j < ARRAY_SIZE; ++j){
            x[j] = (double)j;
        }
    }
    ave_time = total_time / loope;
    printf("Time taken for FWHT: %f ms\n\n", ave_time);

    // Reset data for IFWHT benchmark
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        x[i] = (double)i;
    }

    printf("*** IFWHT ***\n");
    printf("numElements = %lu\n", ARRAY_SIZE);

    total_time = 0.0;
    for (size_t i = 0; i < loope; i++) {
        QueryPerformanceCounter(&StartingTime);
        ifwht_sequency(x, ARRAY_SIZE);
        QueryPerformanceCounter(&EndingTime);
        total_time += ((double)((EndingTime.QuadPart - StartingTime.QuadPart) * 1000000 / Frequency.QuadPart)) / 1000;
        // reset the array before next run.
        for(size_t j = 0; j < ARRAY_SIZE; ++j){
            x[j] = (double)j;
        }
    }
    ave_time = total_time / loope;
    printf("Time taken for IFWHT: %f ms\n\n", ave_time);

    free(x);
    return 0;
}
