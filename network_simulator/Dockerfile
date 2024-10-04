FROM ubuntu:20.04


RUN apt update
RUN apt install -y autoconf automake autotools-dev curl python3 libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev
RUN apt install -y openmpi-bin openmpi-common libtool libtool-bin autoconf python3 python3-dev automake build-essential git
RUN mkdir /home/hwnam
WORKDIR /home/hwnam
RUN git clone https://github.com/sstsimulator/sst-core sst-core-src
RUN git clone https://github.com/sstsimulator/sst-elements sst-elements-src
RUN git clone https://github.com/riscv/riscv-gnu-toolchain
RUN mkdir /opt/riscv
WORKDIR /home/hwnam/riscv-gnu-toolchain
RUN ./configure --prefix=/opt/riscv
RUN make -j16 musl
RUN export PATH=$PATH:/opt/riscv/bin
RUN mkdir /opt/sst
WORKDIR /home/hwnam/sst-core-src
RUN ./autogen.sh
RUN mkdir build
WORKDIR /home/hwnam/sst-core-src/build
RUN ../configure --prefix=/opt/sst CC=`which gcc` CXX=`which g++` MPICC=`which mpicc` MPICXX=`which mpicxx`
RUN make -j16
RUN make install
RUN export PATH=$PATH:/opt/sst/bin
WORKDIR /home/hwnam/sst-elements-src
RUN ./autogen.sh
RUN mkdir build
WORKDIR /home/hwnam/sst-elements-src/build
RUN ../configure --prefix=/opt/sst --with-sst-core=/opt/sst
RUN make -j16
RUN make install
WORKDIR /home/hwnam/
RUN rm -rf riscv-gnu-toolchain
RUN rm -rf sst-elements-src
RUN rm -rf sst-core-src