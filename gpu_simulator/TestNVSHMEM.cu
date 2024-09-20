/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <string>

#include "cuda_fp16.h"
#include "cuda_helper.h"
#include "mpi.h"
#include <nvshmem.h>
#include <nvshmemx.h>

#define PKTSIZE 512
#define NELEM 4096
#define REPEAT 10
#define THREADS_PER_BLOCK 512

__global__ void ring_bcast(half *data, size_t nelem, size_t pkt_size, int root, uint64_t *psync) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    size_t offset = direction * (nelem/2);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = direction == 0 ?
               (mype + 1) % npes : (mype + npes - 1) % npes;

    size_t npackets = (nelem/2 + pkt_size-1) / pkt_size;

    uint64_t *mysync = &psync[direction];
    *mysync = 0;
    
    if (direction == 0 && mype == (root + npes - 1) % npes) return;

    if (direction == 1 && mype == (root + 1) % npes) return;

    for (int idx=0; idx < npackets; idx++){
        half* pos = data + offset + idx*pkt_size;
        int elemcount = idx == npackets-1 ? (nelem/2) - idx*pkt_size : pkt_size;
        if (mype != root)
            nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, idx);
        nvshmemx_put16_block((void*)pos, (void*)pos, elemcount, peer);
        nvshmem_fence();
        nvshmemx_signal_op(mysync, idx+1, NVSHMEM_SIGNAL_SET, peer);
    }
    
}

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define MPI_CHECK(stmt)                                                                         \
    do {                                                                                        \
        int result = (stmt);                                                                    \
        if (MPI_SUCCESS != result) {                                                            \
            fprintf(stderr, "[%s:%d] MPI failed with error %d \n", __FILE__, __LINE__, result); \
            exit(-1);                                                                           \
        }                                                                                       \
    } while (0)

__global__ void simple_shift(int *target, int mype, int npes) {
    int peer = (mype + 1) % npes;
    nvshmem_int_p(target, mype, peer);
}

int main(int c, char *v[]) {
    
    int rank, nranks;
    size_t data_len = NELEM;
    size_t pkt_size = PKTSIZE;
    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;
    int mype, npes, mype_node;
    cudaStream_t stream;

    MPI_CHECK(MPI_Init(&c, &v));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    if (c > 1) data_len = std::stoi(v[1]);//printf("v[1] is %d\t", std::stoi(v[1]));

    if (c > 2) pkt_size = std::stoi(v[2]);//printf("v[2] is %d\n", std::stoi(v[2]));

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    printf("Running benchmark for datalen %d and pktsize %d:",data_len, pkt_size);
    // application picks the device each PE will use
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));
    half *data = (half *)nvshmem_malloc(sizeof(half) * NELEM);
    half *data_h = (half *)malloc(sizeof(half) * NELEM);
    uint64_t *psync = (uint64_t *)nvshmem_calloc(2, sizeof(uint64_t));
    for (int i = 0; i < NELEM; i++) data_h[i] = (half)(mype+1);

    cudaMemcpyAsync(data, data_h, sizeof(half) * NELEM, cudaMemcpyHostToDevice, stream);
    int root = 0;
    dim3 gridDim(2), blockDim(THREADS_PER_BLOCK);
    void *args[] = {&data, &data_len, &pkt_size, &root, &psync};

    nvshmemx_barrier_all_on_stream(stream);
    nvshmemx_collective_launch((const void *)ring_bcast, gridDim, blockDim, args, 0, stream);
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * NELEM, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    /*
    for (size_t i = 0; i < data_len; i++) {
        if ((int)data_h[i] != 1)
            printf("PE %d error, data[%zu] = %d expected data[%zu] = %d\n", mype, i, (int)data_h[i], i,
                   1);
    }*/
    //printf("Warmup done.\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, NULL));
    
    for (int j = 0; j < npes*REPEAT; j++) {
        root = j%npes;
        nvshmemx_collective_launch((const void *)ring_bcast, gridDim, blockDim, args, 0, stream);
    }

    
    nvshmemx_barrier_all_on_stream(stream);
    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / REPEAT / npes;
    double dataSize = 2.0 * data_len;
    double gigaBytePerSec =
        (dataSize * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GB/s, Time= %.3f msec, Size= %.0f KB\n",
        gigaBytePerSec, msecPerMatrixMul, dataSize*1e-3);

    nvshmem_free(data);
    nvshmem_free(psync);
    free(data_h);

    nvshmem_finalize();
    MPI_CHECK(MPI_Finalize());
    return 0;
}

