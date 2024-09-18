NVCC          := nvcc
GENCODE_FLAGS    := -gencode arch=compute_86,code=sm_86
NVCC_FLAGS := -ccbin=mpic++ -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++14

NVCC_LDFLAGS := -ccbin=mpic++ -lcuda -lcudart -lnvToolsExt -lnvidia-ml -lcublas
#NVCC_LDFLAGS += -lnvshmem

all: build

build: TestCUBLAS

TestCUBLAS.o:TestCUBLAS.cu cuda_helper.h
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

TestCUBLAS: TestCUBLAS.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(NVCC_LDFLAGS)

run: build
	./TestCUBLAS

clean:
	rm TestCUBLAS *.o

