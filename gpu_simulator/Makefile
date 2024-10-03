#NVSHMEM_HOME	:= $(NVHPC_HOME)/Linux_x86_64/22.11/comm_libs/nvshmem
NVCC          	:= nvcc
GENCODE_FLAGS	:= -gencode arch=compute_80,code=sm_80
NVCC_FLAGS 		:= -ccbin=mpic++ -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14 -I$(NVSHMEM_HOME)/include #-I$(NCCL_HOME)/include
NVCC_LDFLAGS 	:= -ccbin=mpic++ -lcuda -lcudart -lnvToolsExt -lnvidia-ml -lcublas
NVCC_LDFLAGS 	+= -L$(NVSHMEM_HOME)/lib -lnvshmem -lnccl

all: build

build: TestCUBLAS BenchNVSHMEM TestNVSHMEM TestMesh BenchMesh

TestCUBLAS.o:TestCUBLAS.cu cublas_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestCUBLAS: TestCUBLAS.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

BenchNVSHMEM.o:BenchNVSHMEM.cu RingCollectives.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

BenchNVSHMEM: BenchNVSHMEM.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

TestNVSHMEM.o:TestNVSHMEM.cu RingCollectives.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestNVSHMEM: TestNVSHMEM.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

TestMesh.o:TestMesh.cu MeshCollectives.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestMesh: TestMesh.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

BenchMesh.o:BenchMesh.cu MeshCollectives.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

BenchMesh: BenchMesh.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

BenchNCCL.o:BenchNCCL.cu nccl_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

BenchNCCL: BenchNCCL.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

clean:
	rm TestCUBLAS BenchNVSHMEM TestNVSHMEM *.o

