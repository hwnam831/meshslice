#NVSHMEM_HOME	:= $(NVHPC_HOME)/Linux_x86_64/22.11/comm_libs/nvshmem
NVCC          	:= nvcc
GENCODE_FLAGS	:= -gencode arch=compute_80,code=sm_80
NVCC_FLAGS 		:= -ccbin=mpic++ -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14 -I$(NVSHMEM_HOME)/include
NVCC_LDFLAGS 	:= -ccbin=mpic++ -lcuda -lcudart -lnvToolsExt -lnvidia-ml -lcublas
NVCC_LDFLAGS 	+= -L$(NVSHMEM_HOME)/lib -lnvshmem

all: build

build: TestCUBLAS BenchNVSHMEM TestNVSHMEM

TestCUBLAS.o:TestCUBLAS.cu cuda_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestCUBLAS: TestCUBLAS.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

BenchNVSHMEM.o:BenchNVSHMEM.cu cuda_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

BenchNVSHMEM: BenchNVSHMEM.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

TestNVSHMEM.o:TestNVSHMEM.cu cuda_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestNVSHMEM: TestNVSHMEM.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)


clean:
	rm TestCUBLAS BenchNVSHMEM TestNVSHMEM *.o

