
#include <stdint.h>
#include <string>
#include "MeshCollectives.cuh"

#define PKTSIZE 32768
#define NBLOCKS 16
#define NELEM PKTSIZE*NBLOCKS*2*10
#define THREADS_PER_BLOCK 512
#define REPEAT 10

// mpirun -n 4 datalen pktsize ndev g_y
int main(int c, char *v[]) {
    
    int rank, nranks;
    size_t data_len = NELEM;
    size_t pkt_size = PKTSIZE;
    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;
    int mype, mype_node;
    cudaStream_t stream0, stream1;

    if (c > 1) data_len = std::stoi(v[1]);//printf("v[1] is %d\t", std::stoi(v[1]));

    if (c > 2) pkt_size = std::stoi(v[2]);//printf("v[2] is %d\n", std::stoi(v[2]));
    MPI_CHECK(MPI_Init(&c, &v));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype = nvshmem_my_pe();
    //int npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int myrow = mype / 2;
    int mycol = mype % 2;
    int root = 1;
    int dim0 = 0;
    int dim1 = 1;
    int ndev = 2;
    if (c > 3) ndev = std::stoi(v[3]);

    // application picks the device each PE will use
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));
    half *data0 = (half *)nvshmem_malloc(sizeof(half) * data_len);
    half *data1 = (half *)nvshmem_malloc(sizeof(half) * data_len);

    //half *buf0 = (half *)nvshmem_malloc(sizeof(half) * data_len * ndev);
    //half *buf1 = (half *)nvshmem_malloc(sizeof(half) * data_len * ndev);
    half *data_h = (half *)malloc(sizeof(half) * data_len);
    int g_y = NBLOCKS;
    if (c > 4) g_y = std::stoi(v[4]);
    uint64_t *psync0 = (uint64_t *)nvshmem_calloc(2*g_y, sizeof(uint64_t));
    uint64_t *psync1 = (uint64_t *)nvshmem_calloc(2*g_y, sizeof(uint64_t));
    half *tmp0 = (half *)nvshmem_malloc(sizeof(half) * pkt_size * g_y * 2);
    half *tmp1 = (half *)nvshmem_malloc(sizeof(half) * pkt_size * g_y * 2);
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(0.0);
    
    cudaMemcpyAsync(data0, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(data1, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream1);
    
    
    dim3 gridDim(2, g_y), blockDim(THREADS_PER_BLOCK);
    void *args[] = {&data0, &data_len, &pkt_size, &root, &psync0, &dim0, &ndev};
    printf("\nRunning benchmark for datalen %d, pktsize %d, ndev %d, numCTA %d:\n",data_len, pkt_size, ndev, g_y*2,);

    printf("Benchmarking Broadcast at dim=0, \n");
    nvshmemx_barrier_all_on_stream(stream0);
    nvshmemx_collective_launch((const void *)mesh_bcast, gridDim, blockDim, args, 0, stream0);
    nvshmemx_barrier_all_on_stream(stream0);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, NULL));
    
    for (int j = 0; j < 2*REPEAT; j++) {
        root = j%2;
        nvshmemx_collective_launch((const void *)mesh_bcast, gridDim, blockDim, args, 0, stream0);
        nvshmemx_barrier_all_on_stream(stream0);
    }
    
    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    cudaStreamSynchronize(stream0);

    // Compute and print the performance
    float msecPerRun = msecTotal / REPEAT / 2;
    double dataSize = 2.0 * data_len;
    double gigaBytePerSec =
        (dataSize * 1.0e-9f) / (msecPerRun / 1000.0f);
    printf("Broadcast performance= %.2f GB/s, Time= %.3f msec, Size= %.0f KB\n",
        gigaBytePerSec, msecPerRun, dataSize/1024);

    
    root=0;
    void *args_reduce1[] = {&data1, &tmp1, &data_len, &pkt_size, &root, &psync1, &dim1, &ndev};
    nvshmemx_barrier_all_on_stream(stream1);
    nvshmemx_collective_launch((const void *)mesh_reduce, gridDim, blockDim, args_reduce1, 0, stream1);
    nvshmemx_barrier_all_on_stream(stream1);
    printf("Benchmarking Reduce at dim=1, \n");
    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, NULL));
    
    for (int j = 0; j < 2*REPEAT; j++) {
        root = j%2;
        nvshmemx_collective_launch((const void *)mesh_reduce, gridDim, blockDim, args_reduce1, 0, stream1);
        nvshmemx_barrier_all_on_stream(stream1);
    }
    
    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    cudaStreamSynchronize(stream1);
    // Compute and print the performance
    msecPerRun = msecTotal / REPEAT / 2;
    dataSize = 2.0 * data_len;
    gigaBytePerSec =
        (dataSize * 1.0e-9f) / (msecPerRun / 1000.0f);
    printf("Reduce performance= %.2f GB/s, Time= %.3f msec, Size= %.0f KB\n",
        gigaBytePerSec, msecPerRun, dataSize/1024);

    
    
    printf("Done\n");
    nvshmem_free(data0);
    nvshmem_free(data1);
    nvshmem_free(psync0);
    nvshmem_free(psync1);
    //nvshmem_free(buf0);
    //nvshmem_free(buf1);

    free(data_h);

    nvshmem_finalize();
    MPI_CHECK(MPI_Finalize());
    return 0;
}