NVCC          	:= nvcc
GENCODE_FLAGS	:= -gencode arch=compute_80,code=sm_80
NVCC_FLAGS 		:= -ccbin=mpic++ -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14

NVCC_LDFLAGS 	:= -ccbin=mpic++ -lcuda -lcudart -lnvToolsExt -lnvidia-ml -lcublas
NVCC_LDFLAGS 	+= -lnvshmem

all: build

build: TestCUBLAS TestNVSHMEM

TestCUBLAS.o:TestCUBLAS.cu cuda_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestCUBLAS: TestCUBLAS.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

TestNVSHMEM.o:TestNVSHMEM.cu cuda_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestNVSHMEM: TestNVSHMEM.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

run: build
	mpirun --allow-run-as-root -n 2 TestNVSHMEM

clean:
	rm TestCUBLAS TestNVSHMEM *.o

