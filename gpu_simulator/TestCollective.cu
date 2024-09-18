#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include "cuda_fp16.h"
// Utilities and system includes
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

half* initMatrix(size_t H, size_t W, int val){
    //half* buf_h = (half *)malloc(sizeof(half) * H * W);
    //for (int i=0; i<H*W; i++)
    //    buf_h[i] = val;
    half* buf_d = NULL;
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&buf_d),
                             sizeof(half) * H * W));
    //CUDA_RT_CALL(cudaMemcpy(buf_d, buf_h, sizeof(half) * H * W,
    //                         cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemset(buf_d, val, sizeof(half) * H * W));
    return buf_d;
}

#define M 1280
#define N 512
#define K 1024

int main(){

    half* A = initMatrix(M,K,1);
    half* B = initMatrix(M,K,2);
    half* C = initMatrix(M,N,0);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cudaEvent_t start, stop;

    CUDA_RT_CALL(cublasCreate(&handle));
    // Perform warmup operation with cublas
    CUDA_RT_CALL(cublasHgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, MA,
        K, &alpha, B, N, A, K,
        &beta, C, N));
    half* C_h = (half *)malloc(sizeof(half) * M * N);
    CUDA_RT_CALL(cudaMemcpy(C_h, C, sizeof(half) * M * N, cudaMemcpyDeviceToHost));
    printf("C_h element %f\n", C_h[0]);
    free(C_h);
    CUDA_RT_CALL(cublasDestroy(handle));
    CUDA_RT_CALL(cudaFree(A));
    CUDA_RT_CALL(cudaFree(B));
    return 0;
}

