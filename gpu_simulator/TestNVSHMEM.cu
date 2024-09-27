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
#include <string>
#include "RingCollectives.cuh"

#define PKTSIZE 256
#define NELEM 8192
#define REPEAT 10
#define THREADS_PER_BLOCK 256

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

