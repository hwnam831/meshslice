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

#define PKTSIZE 256
#define NELEM 8192
#define REPEAT 10
#define THREADS_PER_BLOCK 256

__global__ void ring_bcast(half *data, size_t nelem, size_t pkt_size, int root, uint64_t *psync) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t offset = direction * (nelem/2/g_y) + yidx * (nelem/g_y);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = direction == 0 ?
               (mype + 1) % npes : (mype + npes - 1) % npes;

    size_t npackets = (nelem/2/g_y + pkt_size-1) / pkt_size;

    uint64_t *mysync = &psync[direction + 2*yidx];
    *mysync = 0;
    
    if (direction == 0 && mype == (root + npes - 1) % npes) return;

    if (direction == 1 && mype == (root + 1) % npes) return;

    for (int idx=0; idx < npackets; idx++){
        half* pos = data + offset + idx*pkt_size;
        int elemcount = (idx+1)*pkt_size > nelem/2/g_y? (nelem/2/g_y) - idx*pkt_size : pkt_size;
        if (mype != root)
            nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, idx);
        nvshmemx_put16_block((void*)pos, (void*)pos, elemcount, peer);
        nvshmem_quiet();
        nvshmemx_signal_op(mysync, idx+1, NVSHMEM_SIGNAL_SET, peer);
    }
    
}

//In-place reduce algorithm
__global__ void ring_reduce(half *data, half* tmp, size_t nelem, size_t pkt_size, int root, uint64_t *psync) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t offset = direction * (nelem/2/g_y) + yidx * (nelem/g_y);
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int mynext = direction == 0 ?
               (mype + 1) % npes : (mype + npes - 1) % npes;
    int mysrc = direction == 0 ?
                (mype + npes - 1) % npes : (mype + 1) % npes;

    size_t npackets = (nelem/2/g_y + pkt_size-1) / pkt_size;

    uint64_t *mysync = &psync[direction + 2*yidx];

    int source = direction == 0 ?
                (root + 1) % npes : (root + npes - 1) % npes;
    *mysync = (mysrc == source) ? npackets+1 : 0;

    if (mype == source) return;

    half* mytmp = tmp + direction*pkt_size + yidx*2*pkt_size;
    for (int idx=0; idx < npackets; idx++){
        half* pos = data + offset + idx*pkt_size;
        int elemcount = (idx+1)*pkt_size > nelem/2/g_y? (nelem/2/g_y) - idx*pkt_size : pkt_size;
        if(mysrc != source)
            nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, idx);
        nvshmemx_get16_block((void*)mytmp, (void*)pos, elemcount, mysrc);
        nvshmem_quiet();
        half2* pos2 = (half2*)pos;
        half2* mytmp2 = (half2*) mytmp;
        for (int mypos = threadIdx.x; mypos < elemcount/2; mypos += blockDim.x){
            pos2[mypos] = pos2[mypos] + mytmp2[mypos];
        }
        __syncthreads();
        nvshmemx_signal_op(mysync, idx+1, NVSHMEM_SIGNAL_SET, mynext);
    }
    
}

__global__ void ring_allgather(half *shard, half *buf, size_t nelem, uint64_t *psync) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t elemcount = nelem/2/g_y;
    size_t offset = (direction + 2*yidx) * elemcount;
    
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = direction == 0 ?
               (mype + 1) % npes : (mype + npes - 1) % npes;

    uint64_t *mysync = &psync[direction + 2*yidx];
    *mysync = 0;
    //First, my shard is written to the buffer
    half2* mypos = (half2*)&buf[nelem*mype + offset];
    half2* shard2 = (half2*)&shard[offset];
    for (int idx = threadIdx.x; idx < elemcount/2; idx += blockDim.x){
        mypos[idx] = shard2[idx];
    }
    __syncthreads();
    for (int iter=0; iter < npes-1; iter++){
        mypos = (half2*)&buf[nelem*mype + offset];
        nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_EQ, iter);
        nvshmemx_put32_block((void*)mypos, (void*)mypos, elemcount/2, peer);
        nvshmem_quiet();
        nvshmemx_signal_op(mysync, iter+1, NVSHMEM_SIGNAL_SET, peer);
        mype = direction == 0 ?
            (mype + npes - 1) % npes : (mype + 1) % npes;
    }
    
}

__global__ void ring_reducescatter(half *shard, half *buf, size_t nelem, uint64_t *psync) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t elemcount = nelem/2/g_y;
    size_t offset = (direction + 2*yidx) * elemcount;
    
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = direction == 0 ?
               (mype + 1) % npes : (mype + npes - 1) % npes;
    int mynext = direction == 1 ?
               (mype + 1) % npes : (mype + npes - 1) % npes;

    uint64_t *mysync = &psync[direction + 2*yidx];
    *mysync = 0;
    //First, my shard is written to the buffer
    half* mypos = &buf[nelem*mype + offset];
    half* myshard = &shard[offset];
    half2* myshard2 = (half2*)myshard;
    half2* mypos2 = (half2*)mypos;
    __syncthreads();
    int curpe = peer;
    for (int iter=0; iter < npes-1; iter++){
        curpe = direction == 0 ?
            (curpe + 1) % npes : (curpe + npes - 1) % npes;
        mypos = &buf[nelem*curpe + offset];
        nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_EQ, iter);
        nvshmemx_get16_block((void*)myshard, (void*)mypos, elemcount, peer);
        nvshmem_quiet();
        mypos2 = (half2*)mypos;
        for (int idx = threadIdx.x; idx < elemcount/2; idx += blockDim.x){
            mypos2[idx] = mypos2[idx] + myshard2[idx];
        }
        __syncthreads();
        nvshmemx_signal_op(mysync, iter+1, NVSHMEM_SIGNAL_SET, mynext);
        
    }
    
    mypos = &buf[nelem*mype + offset];
    mypos2 = (half2*)mypos;
    for (int idx = threadIdx.x; idx < elemcount/2; idx += blockDim.x){
        myshard2[idx] = mypos2[idx];
    }
    __syncthreads();
    
}

#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

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

#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

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
    half *data = (half *)nvshmem_malloc(sizeof(half) * data_len);
    half *data_h = (half *)malloc(sizeof(half) * data_len);
    const int g_y = 4;
    uint64_t *psync = (uint64_t *)nvshmem_calloc(2*g_y, sizeof(uint64_t));
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+(1.0/PKTSIZE)*i);

    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);
    int root = 0;
    
    dim3 gridDim(2, g_y), blockDim(THREADS_PER_BLOCK);
    void *args[] = {&data, &data_len, &pkt_size, &root, &psync};
    
    nvshmemx_barrier_all_on_stream(stream);
    nvshmemx_collective_launch((const void *)ring_bcast, gridDim, blockDim, args, 0, stream);
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    for (size_t i = 0; i < data_len; i++) {
        float diff = (float)data_h[i] - (1.0/PKTSIZE)*i;
        if (diff > 0.1 || diff < -0.1)
            printf("Broadcast PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, (float)data_h[i], i,
                (1.0/PKTSIZE)*i);
    }
    printf("Broadcast test done.\n");
    
    printf("Running reduce test.\n");
    cudaStream_t stream2;
    CUDA_CHECK(cudaStreamCreate(&stream2));
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+1+(1.0/PKTSIZE)*i);
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream2);

    half *tmp_data = (half *)nvshmem_malloc(sizeof(half) * g_y * 2*pkt_size);

    void *args2[] = {&data, &tmp_data, &data_len, &pkt_size, &root, &psync};

    nvshmemx_barrier_all_on_stream(stream2);
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)ring_reduce, gridDim, blockDim, args2, 0, stream2));
    //checkKernelErrors((ring_reduce<<<2,THREADS_PER_BLOCK>>>(data,tmp_data,data_len,pkt_size,root,psync)));
    nvshmemx_barrier_all_on_stream(stream2);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream2);
    
    if (mype == root){
        for (size_t i = 0; i < data_len; i++) {
            float sum = (npes+1)*npes/2 + (1.0/PKTSIZE)*i * npes;
            float diff = (float)data_h[i] - sum;
            if (diff > 0.1 || diff < -0.1)
                printf("Reduce PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, (float)data_h[i], i,
                    sum);
        }
    }
    printf("Reduce test done.\n");
    printf("Running Allgather test.\n");
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+1+(1.0/PKTSIZE)*i);
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream2);
    half *buf = (half *)nvshmem_malloc(sizeof(half) * data_len*npes);
    half *buf_h = (half *)malloc(sizeof(half) * data_len*npes);
    nvshmemx_barrier_all_on_stream(stream2);
    void *args3[] = {&data, &buf, &data_len, &psync};
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)ring_allgather, gridDim, blockDim, args3, 0, stream2));
    nvshmemx_barrier_all_on_stream(stream2);
    cudaMemcpyAsync(buf_h, buf, sizeof(half) * data_len*npes, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream2);

    for (int pecount = 0; pecount < npes; pecount++){
        for (size_t i = 0; i < data_len; i++) {
            float diff = (float)buf_h[pecount*data_len + i] - (pecount+1+(1.0/PKTSIZE)*i);
            if (diff > 0.1 || diff < -0.1){
                printf("Allgather PE %d error, buf[%zu] = %f expected buf[%zu] = %f\n", mype, pecount*data_len + i, 
                (float)buf_h[pecount*data_len + i], i, (pecount+1+(1.0/PKTSIZE)*i));
                break;
            }

        }
    }
    printf("Allgather test done\n");

    printf("Running Reducescatter test.\n");
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)ring_reducescatter, gridDim, blockDim, args3, 0, stream2));
    nvshmemx_barrier_all_on_stream(stream2);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(buf_h, buf, sizeof(half) * data_len*npes, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream2);
    
    for (int i = 0; i < data_len; i++) {
        float diff = (float)data_h[i] - ((mype+1+(1.0/PKTSIZE)*i)*npes);
        if (diff > 0.1 || diff < -0.1) {
            printf("Reducescatter PE %d error, data[%zu] = %f expected data[%zu] = %f\n",
            mype, i, (float)data_h[i], i, (mype+1+(1.0/PKTSIZE)*i)*npes);
            //break;
        }
    }
    printf("Reducescatter test done\n");
    nvshmem_free(tmp_data);
    nvshmem_free(data);
    nvshmem_free(psync);
    nvshmem_free(buf);

    free(data_h);
    free(buf_h);

    nvshmem_finalize();
    MPI_CHECK(MPI_Finalize());
    return 0;
}

