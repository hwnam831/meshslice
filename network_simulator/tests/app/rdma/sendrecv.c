// Copyright 2009-2024 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2024, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#include <stdio.h>
#include <stdlib.h>
#include <rdma.h>
#include <unistd.h>
#include <strings.h>
#include <inttypes.h>

//#define BUF_SIZE 250000
#define BUF_SIZE 20480*1024

int32_t send_buf[BUF_SIZE];
int32_t recv_buf[BUF_SIZE];

int main( int argc, char* argv[] ) {

	rdma_init();

	int myNode = rdma_getMyNode();
	int numNodes = rdma_getNumNodes();
	int peer = (myNode + 1) % numNodes;
	printf("%s() myNode=%d numNodes=%d\n",__func__,myNode,numNodes);
		
	int msgCq = rdma_create_cq( );

	int cq = rdma_create_cq( );

	int rq = rdma_create_rq( 0xbeef, msgCq );

    printf("%s() call barrier\n",__func__);
	/*
	for ( int i = 0; i < BUF_SIZE; i++ ) {
		send_buf[i] = myNode;
	}*/
	
    rdma_barrier();

	
	rdma_recv_post( (void*)recv_buf, BUF_SIZE * sizeof(int32_t), rq, (Context) recv_buf );
	

	rdma_send_post( (void*)send_buf, BUF_SIZE * sizeof(int32_t), peer, 0, 0xbeef, cq, 0xf00dbeef ); 
	RdmaCompletion comp;
	rdma_read_comp( cq, &comp, 1 );

	rdma_read_comp( msgCq, &comp, 1 );
	/*
	for ( int i = 0; i < BUF_SIZE; i++ ) {
		if ( recv_buf[i] != (myNode + numNodes - 1)%numNodes ) {
			printf("Error: %p index=%d != %d\n",&recv_buf[i],i,recv_buf[i]);
			break;
		}
	}*/

	rdma_fini();
	printf("%s() returning\n",__func__);
	return 0;
}
