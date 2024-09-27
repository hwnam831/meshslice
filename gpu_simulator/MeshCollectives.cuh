/**
This collectives will only work in a 4gpu node.
*/

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <string>

#include "cuda_fp16.h"
#include "mpi.h"
#include <nvshmem.h>
#include <nvshmemx.h>

// dim: 0 is inter-row (same col), 1 is inter-column (same row)
// ndev: number of chips in col/row to simulate
__global__ void 
mesh_bcast(half *data, size_t nelem, size_t pkt_size,
            int root, uint64_t *psync, int dim, int ndev) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t offset = direction * (nelem/2/g_y) + yidx * (nelem/g_y);

    int mype = nvshmem_my_pe();
    int myrow = mype / 2;
    int mycol = mype % 2;
    int peer = dim == 0?
               ((myrow+1)%2)*2 + mycol : myrow*2 + (mycol+1)%2;
    int local_id = dim == 0?: myrow: mycol;
    size_t npackets = (nelem/2/g_y + pkt_size-1) / pkt_size;

    uint64_t *mysync = &psync[direction + 2*yidx];
    *mysync = local_id == root? 1 : 0;
    
    // circulate the packets (ndev-2) times.
    // This simulates the time that the first packet arrives to the last node.
    half* pos = data + offset;
    for (int repeat = 0; repeat < (ndev-2)/2; repeat++){
        nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, repeat);
        nvshmemx_put16_block((void*)pos, (void*)pos, pkt_size, peer);
        nvshmem_quiet();
        nvshmemx_signal_op(mysync, 1, NVSHMEM_SIGNAL_ADD, peer);
    }

    if (local_id != root) return; 

    // Actual broadcast
    for (int idx=0; idx < npackets; idx++){
        pos = data + offset + idx*pkt_size;
        int elemcount = (idx+1)*pkt_size > nelem/2/g_y? (nelem/2/g_y) - idx*pkt_size : pkt_size;
        nvshmemx_put16_block((void*)pos, (void*)pos, elemcount, peer);
        nvshmem_quiet();
        nvshmemx_signal_op(mysync, idx+1, NVSHMEM_SIGNAL_SET, peer);
    }
}

//In-place reduce algorithm
__global__ void 
mesh_reduce(half *data, half* tmp, size_t nelem,size_t pkt_size,
            int root, uint64_t *psync, int dim, int ndev) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t offset = direction * (nelem/2/g_y) + yidx * (nelem/g_y);
    int mype = nvshmem_my_pe();
    int myrow = mype / 2;
    int mycol = mype % 2;
    int peer = dim == 0?
               ((myrow+1)%2)*2 + mycol : myrow*2 + (mycol+1)%2;
    int local_id = dim == 0?: myrow: mycol;

    size_t npackets = (nelem/2/g_y + pkt_size-1) / pkt_size;

    uint64_t *mysync = &psync[direction + 2*yidx];

    *mysync = (local_id == root) ? 1 : 0;

    // circulate the packets (ndev-2) times.
    // This simulates the time that the first packet arrives to the last node.
    half* pos = data + offset;
    half* mytmp = tmp + direction*pkt_size + yidx*2*pkt_size;
    for (int repeat = 0; repeat < (ndev-2)/2; repeat++){
        nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, repeat);
        nvshmemx_get16_block((void*)mytmp, (void*)pos, pkt_size, peer);
        nvshmem_quiet();
        half2* pos2 = (half2*)pos;
        half2* mytmp2 = (half2*) mytmp;
#pragma unroll
        for (int mypos = threadIdx.x; mypos < pkt_size/2; mypos += blockDim.x){
            mytmp2[mypos] += pos[mypos];
        }
        nvshmemx_signal_op(mysync, 1, NVSHMEM_SIGNAL_ADD, peer);
    }
    if (local_id 1= root) return;

    
    for (int idx=0; idx < npackets; idx++){
        pos = data + offset + idx*pkt_size;
        int elemcount = (idx+1)*pkt_size > nelem/2/g_y? (nelem/2/g_y) - idx*pkt_size : pkt_size;
        nvshmemx_get16_block((void*)mytmp, (void*)pos, elemcount, peer);
        nvshmem_quiet();
        half2* pos2 = (half2*)pos;
        half2* mytmp2 = (half2*) mytmp;
#pragma unroll
        for (int mypos = threadIdx.x; mypos < elemcount/2; mypos += blockDim.x){
            pos2[mypos] += mytmp2[mypos];
        }
        nvshmemx_signal_op(mysync, idx+1, NVSHMEM_SIGNAL_SET, peer);
    }
    
}

__global__ void
mesh_allgather(half *shard, half *buf, size_t nelem,
                uint64_t *psync, int dim, int ndev) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t elemcount = nelem/2/g_y;
    size_t offset = (direction + 2*yidx) * elemcount;
    
    int mype = nvshmem_my_pe();
    int myrow = mype / 2;
    int mycol = mype % 2;
    int peer = dim == 0?
               ((myrow+1)%2)*2 + mycol : myrow*2 + (mycol+1)%2;
    int local_id = dim == 0?: myrow: mycol;

    uint64_t *mysync = &psync[direction + 2*yidx];
    *mysync = 1;
    //First, my shard is written to the buffer
    half2* mypos = (half2*)&buf[nelem*local_id + offset];
    half2* shard2 = (half2*)&shard[offset];
    for (int idx = threadIdx.x; idx < elemcount/2; idx += blockDim.x){
        mypos[idx] = shard2[idx];
    }
    __syncthreads();
    int mydev = local_id;
    for (size_t iter=0; iter < ndev-1; iter++){
        
        mypos = (half2*)&buf[nelem*mydev + offset];
        nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, iter);
        nvshmemx_put32_block((void*)mypos, (void*)mypos, elemcount/2, peer);
        nvshmem_quiet();
        nvshmemx_signal_op(mysync, iter+2, NVSHMEM_SIGNAL_SET, peer);
        mydev = direction == 0 ?
            (mydev + ndev - 1) % ndev : (mydev + 1) % ndev;
    }
    
}

__global__ void 
mesh_reducescatter(half *shard, half *buf, size_t nelem,
                    uint64_t *psync, int dim, int ndev) {
    //Bidirectional algorithm. First CTA sends rightwards and second CTA sends in opposite direction. 
    int direction = blockIdx.x % 2;
    int yidx = blockIdx.y;
    int g_y = gridDim.y;
    size_t elemcount = nelem/2/g_y;
    size_t offset = (direction + 2*yidx) * elemcount;
    
    int mype = nvshmem_my_pe();
    int myrow = mype / 2;
    int mycol = mype % 2;
    int peer = dim == 0?
               ((myrow+1)%2)*2 + mycol : myrow*2 + (mycol+1)%2;
    int local_id = dim == 0?: myrow: mycol;

    uint64_t *mysync = &psync[direction + 2*yidx];
    *mysync = 0;
    
    half* mypos = &buf[nelem*local_id + offset];
    half* myshard = &shard[offset];
    half2* myshard2 = (half2*)myshard;
    half2* mypos2 = (half2*)mypos;
    __syncthreads();
    int curpe = direction == 0 ?
        (local_id+1)%ndev : (local_id + ndev - 1) % ndev;
    for (int iter=0; iter < ndev-1; iter++){
        curpe =  direction == 0 ?
            (curpe+1)%ndev : (curpe + ndev - 1) % ndev;
        mypos = &buf[nelem*curpe + offset];
        nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_EQ, iter);
        nvshmemx_get32_block((void*)myshard, (void*)mypos, elemcount/2, peer);
        nvshmem_quiet();
        mypos2 = (half2*)mypos;
        for (int idx = threadIdx.x; idx < elemcount/2; idx += blockDim.x){
            mypos2[idx] = mypos2[idx] + myshard2[idx];
        }
        __syncthreads();
        nvshmemx_signal_op(mysync, iter+1, NVSHMEM_SIGNAL_SET, mynext);
    }
    
    mypos = &buf[nelem*local_id + offset];
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