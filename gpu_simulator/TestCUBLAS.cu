#include <mpi.h>
//#include <nvshmem.h>
//#include <nvshmemx.h>

#include <cstdio>
#include <iostream>

#include <cstdlib>
#include "cuda_fp16.h"
// Utilities and system includes
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cublas_helper.h"

#define M 2560
#define N 2048
#define K 3072
#define REPEAT 10

int main(){

    half* A = initMatrix(M,K,1);
    half* B = initMatrix(N,K,2);
    half* C = initMatrix(M,N,0);
    const half alpha = 1.0f;
    const half beta = 0.0f;
    cublasHandle_t handle;
    cudaEvent_t start, stop;
    
    checkCudaErrors(cublasCreate(&handle));
    // Perform warmup operation with cublas
    cudaDeviceSynchronize();
    checkCudaErrors(cublasHgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M,
        K, &alpha, B, N, A, K,
        &beta, C, N));
    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    CUDA_RT_CALL(cudaEventCreate(&start));
    CUDA_RT_CALL(cudaEventCreate(&stop));

    // Record the start event
    CUDA_RT_CALL(cudaEventRecord(start, NULL));

    for (int j = 0; j < REPEAT; j++) {
      // note cublas is column primary!
      // need to transpose the order
      checkCudaErrors(cublasHgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M,
        K, &alpha, B, N, A, K,
        &beta, C, N));
    }

    printf("done.\n");

    // Record the stop event
    CUDA_RT_CALL(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CUDA_RT_CALL(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CUDA_RT_CALL(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / REPEAT;
    double flopsPerMatrixMul = 2.0 * (double)M *
                               (double)N *
                               (double)K;
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    half* C_h = (half *)malloc(sizeof(half) * M * N);
    CUDA_RT_CALL(cudaMemcpy(C_h, C, sizeof(half) * M * N, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    printf("C_h element %f\n", (float)C_h[0]);

    free(C_h);

    checkCudaErrors(cublasDestroy(handle));
    CUDA_RT_CALL(cudaFree(A));
    CUDA_RT_CALL(cudaFree(B));
    return 0;
}

