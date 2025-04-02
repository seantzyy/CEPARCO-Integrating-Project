
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>

cudaError_t FHWTCuda();

__global__ void firstStageKernel(double* x, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N / 2) {
        double temp1 = x[2 * i];
        double temp2 = x[2 * i + 1];
        x[2 * i] = temp1 + temp2;
        x[2 * i + 1] = temp1 - 2 * temp2;
    }
}

__global__ void secondStagesKernel(double* x, double* y, size_t N, size_t M) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread index

    if (idx < N / (2 * M)) { // Ensure the index is within bounds
        size_t j = idx * 2 * M; // Calculate position in the array

        y[j] = x[j] + x[j + M];
        y[j + 1] = x[j] - x[j + M];
        y[j + 2] = x[j + 1] - x[j + 1 + M];
        y[j + 3] = x[j + 1] + x[j + 1 + M];
    }
}



__global__ void ifwht_kernel(double* data, size_t N) {
    __syncthreads();
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        data[index] /= (double)N;
    }
}

int main()
{
    cudaError_t cudaStatus = FHWTCuda();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FWHTCuda failed!");
        return 1;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t FHWTCuda()
{
    LARGE_INTEGER StartingTime, EndingTime;
    LARGE_INTEGER Frequency;
    QueryPerformanceFrequency(&Frequency);
    double total_time, ave_time;
    double* x, *d_x, *output, *dy;
    const size_t ARRAY_SIZE = 1 << 20;
    const size_t ARRAY_BYTES = ARRAY_SIZE * sizeof(double);
    const size_t loope = 5;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    int device = -1;
    cudaGetDevice(&device);
    cudaMallocManaged(&x, ARRAY_BYTES);
    cudaMallocManaged(&output, ARRAY_BYTES);
    cudaMalloc((void**)&dy, ARRAY_SIZE * sizeof(double));
    cudaMalloc((void**)&d_x, ARRAY_SIZE * sizeof(double));
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMallocManaged(&x, ARRAY_BYTES);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = (double)i;
    }


    size_t numThreads = 1024;
    size_t numBlocks = (ARRAY_SIZE + numThreads - 1) / numThreads;
    printf("*** FWHT ***\n");
    printf("numElements = %lu\n", ARRAY_SIZE);
    printf("numBlocks = %lu, numThreads = %lu \n", numBlocks, numThreads);

    cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    total_time = 0.0;

    for (size_t i = 0; i < loope; i++) {
        QueryPerformanceCounter(&StartingTime);
        cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        firstStageKernel <<<numBlocks, numThreads >>> (d_x, ARRAY_SIZE);
        cudaDeviceSynchronize();

        size_t M = 2;
        for (size_t nStage = 2; nStage <= log2(ARRAY_SIZE); nStage++) {
            secondStagesKernel <<<numBlocks, numThreads >>> (d_x, dy, ARRAY_SIZE, M);
            cudaDeviceSynchronize();

            // Copy results back to d_x for the next stage
            cudaMemcpy(d_x, dy, ARRAY_SIZE * sizeof(double), cudaMemcpyDeviceToDevice);
            M *= 2;
        }
        QueryPerformanceCounter(&EndingTime);
        total_time += ((double)((EndingTime.QuadPart - StartingTime.QuadPart) * 1000000 / Frequency.QuadPart)) / 1000;
    }
    ave_time = total_time / loope;
    printf("Time taken for FWHT: %f ms\n\n", ave_time);


    printf("*** IFWHT ***\n");
    printf("numBlocks = %lu, numThreads = %lu \n", numBlocks, numThreads);

    total_time = 0.0;
    for (size_t i = 0; i < loope; i++) {
        QueryPerformanceCounter(&StartingTime);
        cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        firstStageKernel <<<numBlocks, numThreads >>> (d_x, ARRAY_SIZE);
        cudaDeviceSynchronize();

        size_t M = 2;
        for (size_t nStage = 2; nStage <= log2(ARRAY_SIZE); nStage++) {
            secondStagesKernel <<<numBlocks, numThreads >>> (d_x, dy, ARRAY_SIZE, M);
            cudaDeviceSynchronize();

            // Copy results back to d_x for the next stage
            cudaMemcpy(d_x, dy, ARRAY_SIZE * sizeof(double), cudaMemcpyDeviceToDevice);
            M *= 2;
        }
        ifwht_kernel <<<numBlocks, numThreads >>> (d_x, ARRAY_SIZE);
        QueryPerformanceCounter(&EndingTime);
        total_time += ((double)((EndingTime.QuadPart - StartingTime.QuadPart) * 1000000 / Frequency.QuadPart)) / 1000;
    }
    ave_time = total_time / loope;
    printf("Time taken for IFWHT: %f ms\n\n", ave_time);
 
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }


Error:
    cudaFree(x);
    cudaFree(d_x);
    cudaFree(output);

    return cudaStatus;
}
