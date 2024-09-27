#include <cuda.h>
#include "cuda_fp16.h"
#include "mpi.h"
#include <nvshmem.h>
#include <nvshmemx.h>

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
        nvshmemx_put32_block((void*)pos, (void*)pos, elemcount/2, peer);
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
#pragma unroll
        for (int mypos = threadIdx.x; mypos < elemcount/2; mypos += blockDim.x){
            pos2[mypos] += mytmp2[mypos];
        }
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
    *mysync = 1;
    //First, my shard is written to the buffer
    half2* mypos = (half2*)&buf[nelem*mype + offset];
    half2* shard2 = (half2*)&shard[offset];
    for (int idx = threadIdx.x; idx < elemcount/2; idx += blockDim.x){
        mypos[idx] = shard2[idx];
    }
    __syncthreads();
    for (size_t iter=0; iter < npes-1; iter++){
        mypos = (half2*)&buf[nelem*mype + offset];
        nvshmem_signal_wait_until(mysync, NVSHMEM_CMP_GT, iter);
        nvshmemx_put32_block((void*)mypos, (void*)mypos, elemcount/2, peer);
        nvshmem_quiet();
        nvshmemx_signal_op(mysync, iter+2, NVSHMEM_SIGNAL_SET, peer);
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
        nvshmemx_get32_block((void*)myshard, (void*)mypos, elemcount/2, peer);
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