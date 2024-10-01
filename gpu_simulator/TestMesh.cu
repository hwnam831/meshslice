
#include <stdint.h>
#include <string>
#include "MeshCollectives.cuh"

#define PKTSIZE 256
#define NELEM 8192
#define THREADS_PER_BLOCK 128

int main(int c, char *v[]) {
    
    int rank, nranks;
    size_t data_len = NELEM;
    size_t pkt_size = PKTSIZE;
    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;
    int mype, mype_node;
    cudaStream_t stream;

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
    int dim = 0;
    int ndev = 2;

    // application picks the device each PE will use
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));
    half *data = (half *)nvshmem_malloc(sizeof(half) * data_len);
    half *data_h = (half *)malloc(sizeof(half) * data_len);
    const int g_y = 4;
    uint64_t *psync = (uint64_t *)nvshmem_calloc(2*g_y, sizeof(uint64_t));
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+(1.0/PKTSIZE)*i);
    
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);
    
    
    dim3 gridDim(2, g_y), blockDim(THREADS_PER_BLOCK);
    void *args[] = {&data, &data_len, &pkt_size, &root, &psync, &dim, &ndev};
    

    printf("Testing Broadcast at dim=0\n");
    nvshmemx_barrier_all_on_stream(stream);
    nvshmemx_collective_launch((const void *)mesh_bcast, gridDim, blockDim, args, 0, stream);
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    for (size_t i = 0; i < data_len; i++) {
        float diff = (float)data_h[i] - (2 + mycol + (1.0/PKTSIZE)*i);
        if (diff > 0.1 || diff < -0.1)
            printf("Broadcast (dim=0) PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, (float)data_h[i], i,
            (2 + mycol + (1.0/PKTSIZE)*i));
    }


    printf("Testing Broadcast at dim=1\n");
    dim=1;
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+(1.0/PKTSIZE)*i);
    
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);
    nvshmemx_barrier_all_on_stream(stream);
    nvshmemx_collective_launch((const void *)mesh_bcast, gridDim, blockDim, args, 0, stream);
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    for (size_t i = 0; i < data_len; i++) {
        float diff = (float)data_h[i] - (1 + myrow*2 + i*1.0/PKTSIZE);
        if (diff > 0.1 || diff < -0.1)
            printf("Broadcast (dim=1) PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, (float)data_h[i], i,
            (1 + myrow*2 + i*1.0/PKTSIZE));
    }

    
    printf("Running reduce test. at dim=0\n");
    root=0;
    dim=0;
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+(1.0/PKTSIZE)*i);
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);

    half *tmp_data = (half *)nvshmem_malloc(sizeof(half) * g_y * 2*pkt_size);
    int ndev2 = 8;
    void *args2[] = {&data, &tmp_data, &data_len, &pkt_size, &root, &psync, &dim, &ndev2};

    nvshmemx_barrier_all_on_stream(stream);
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)mesh_reduce, gridDim, blockDim, args2, 0, stream));
    //checkKernelErrors((ring_reduce<<<2,THREADS_PER_BLOCK>>>(data,tmp_data,data_len,pkt_size,root,psync)));
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    if (myrow == root){
        for (size_t i = 0; i < data_len; i++) {
            float sum = (mycol*2 + 2 + (2.0/PKTSIZE)*i);
            float diff = (float)data_h[i] - sum;
            if (diff > 0.1 || diff < -0.1)
                printf("Reduce PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, (float)data_h[i], i,
                    sum);
        }
    }


    printf("Running reduce test. at dim=1\n");
    dim=1;
    
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+(1.0/PKTSIZE)*i);
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);

    nvshmemx_barrier_all_on_stream(stream);
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)mesh_reduce, gridDim, blockDim, args2, 0, stream));
    //checkKernelErrors((ring_reduce<<<2,THREADS_PER_BLOCK>>>(data,tmp_data,data_len,pkt_size,root,psync)));
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    if (mycol == root){
        for (size_t i = 0; i < data_len; i++) {
            float sum = (myrow*2*2 + 1 + (2.0/PKTSIZE)*i);
            float diff = (float)data_h[i] - sum;
            if (diff > 0.1 || diff < -0.1)
                printf("Reduce PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, (float)data_h[i], i,
                    sum);
        }
    }

    
    printf("Running Allgather test. at dim=0\n");
    dim=0;
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+(1.0/PKTSIZE)*i);
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);
    half *buf = (half *)nvshmem_malloc(sizeof(half) * data_len*ndev);
    half *buf_h = (half *)malloc(sizeof(half) * data_len*ndev);
    nvshmemx_barrier_all_on_stream(stream);
    void *args3[] = {&data, &buf, &data_len, &psync, &dim, &ndev};
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)mesh_allgather, gridDim, blockDim, args3, 0, stream));
    nvshmemx_barrier_all_on_stream(stream);
    cudaMemcpyAsync(buf_h, buf, sizeof(half) * data_len*ndev, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (int pecount = 0; pecount < ndev; pecount++){
        for (size_t i = 0; i < data_len; i++) {
            float diff = (float)buf_h[pecount*data_len + i] - (mycol + 2*pecount +(1.0/PKTSIZE)*i);
            if (diff > 0.1 || diff < -0.1){
                printf("Allgather PE %d error, buf[%zu] = %f expected buf[%zu] = %f\n", mype, pecount*data_len + i, 
                (float)buf_h[pecount*data_len + i], i, (mycol + 2*pecount +(1.0/PKTSIZE)*i));
                break;
            }

        }
    }

    printf("Running Reducescatter test at dim=0.\n");
    dim=0;
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)mesh_reducescatter, gridDim, blockDim, args3, 0, stream));
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < data_len; i++) {
        float sum = (mype*2 + (2.0/PKTSIZE)*i);
        float diff = (float)data_h[i] - sum;
        if (diff > 0.1 || diff < -0.1) {
            printf("Reducescatter PE %d error, data[%zu] = %f expected data[%zu] = %f\n",
            mype, i, (float)data_h[i], i, sum);
            //break;
        }
    }

    printf("Running Allgather test. at dim=1\n");
    dim=1;
    for (int i = 0; i < data_len; i++) data_h[i] = (half)(mype+(1.0/PKTSIZE)*i);
    cudaMemcpyAsync(data, data_h, sizeof(half) * data_len, cudaMemcpyHostToDevice, stream);

    nvshmemx_barrier_all_on_stream(stream);
    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)mesh_allgather, gridDim, blockDim, args3, 0, stream));
    nvshmemx_barrier_all_on_stream(stream);
    cudaMemcpyAsync(buf_h, buf, sizeof(half) * data_len*ndev, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    for (int pecount = 0; pecount < ndev; pecount++){
        for (size_t i = 0; i < data_len; i++) {
            float diff = (float)buf_h[pecount*data_len + i] - (2*myrow + pecount +(1.0/PKTSIZE)*i);
            if (diff > 0.1 || diff < -0.1){
                printf("Allgather PE %d error, buf[%zu] = %f expected buf[%zu] = %f\n", mype, pecount*data_len + i, 
                (float)buf_h[pecount*data_len + i], i, (mycol + 2*pecount +(1.0/PKTSIZE)*i));
                break;
            }

        }
    }

    printf("Running Reducescatter test at dim=1.\n");

    NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)mesh_reducescatter, gridDim, blockDim, args3, 0, stream));
    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(data_h, data, sizeof(half) * data_len, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    for (int i = 0; i < data_len; i++) {
        float sum = (mype*2 + (2.0/PKTSIZE)*i);
        float diff = (float)data_h[i] - sum;
        if (diff > 0.1 || diff < -0.1) {
            printf("Reducescatter PE %d error, data[%zu] = %f expected data[%zu] = %f\n",
            mype, i, (float)data_h[i], i, sum);
            //break;
        }
    }    
    
    printf("Done\n");
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