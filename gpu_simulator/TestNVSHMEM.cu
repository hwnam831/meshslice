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

#include "cuda_fp16.h"
#include "cuda_helper.h"
#include "mpi.h"
#include <nvshmem.h>
#include <nvshmemx.h>

#define PKTSIZE 8
#define NELEM 1024

__global__ void ring_bcast(half *data, size_t nelem, int root, uint64_t *psync) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    size_t offset = direction * (nelem/2);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = direction == 0 ?
               (mype + 1) % npes : (mype + npes - 1) % npes;

    size_t npackets = (nelem/2 + PKTSIZE-1) / PKTSIZE;

    uint64_t *mysync = &psync[direction];
    *mysync = 0;
    
    if (direction == 0 && mype == (root + npes - 1) % npes) return;

    if (direction == 1 && mype == (root + 1) % npes) return;

    for (int idx=0; idx < npackets; idx++){
        half* pos = data + offset + idx*PKTSIZE;
        int elemcount = idx == npackets-1 ? (nelem/2) - idx*PKTSIZE : PKTSIZE;
        if (mype != root)
            nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, idx);
        nvshmem_put16((void*)pos, (void*)pos, elemcount, peer);
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
    
    size_t data_len = 32;
    cudaStream_t stream;

    nvshmem_init();

    int mype = nvshmem_my_pe();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    half *data = (half *)nvshmem_malloc(sizeof(half) * data_len);
    half *data_h = (half *)malloc(sizeof(half) * data_len);
    uint64_t *psync = (uint64_t *)nvshmem_calloc(2, sizeof(uint64_t));

    for (size_t i = 0; i < data_len; i++) data_h[i] = (half)(mype + i);

    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);

    int root = 0;
    dim3 gridDim(2), blockDim(1);
    void *args[] = {&data, &data_len, &root, &psync};

    nvshmemx_barrier_all_on_stream(stream);
    nvshmemx_collective_launch((const void *)ring_bcast, gridDim, blockDim, args, 0, stream);
    nvshmemx_barrier_all_on_stream(stream);
    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < data_len; i++) {
        if ((int)data_h[i] != (int)i)
            printf("PE %d error, data[%zu] = %d expected data[%zu] = %d\n", mype, i, (int)data_h[i], i,
                   (int)i);
    }

    nvshmem_free(data);
    nvshmem_free(psync);
    free(data_h);

    nvshmem_finalize();
    return 0;
}

