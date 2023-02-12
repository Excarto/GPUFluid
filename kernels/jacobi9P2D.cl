
// Single iteration of a Jacobi solver using a 9-point stencil.

__kernel
void jacobi9P2D(__global const float* bdat,
          __global float* xdatIn,
		  __global float* xdatOut,
		  const float alpha,
		  const float beta){
	
	uint blockX = get_group_id(0);
	uint blockY = get_group_id(1);
	uint localId = GROUP_SIZE - get_local_id(0) - 1;
	
	uint blockCellX = blockX*BLOCK_SIZE;
	uint blockCellY = blockY*BLOCK_SIZE;
	
	uint cell = localId*CELLS_PER_ITEM;
	uint cellY = (cell*ITEM_DIV_CONST) >> DIV_POW;
	uint cellX = cell - cellY*BLOCK_SIZE;
	cellX++;
	cellY++;
	
	// OPERATE AND OUTPUT
	
	uint datIndex = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float sum = alpha*bdat[datIndex];
		sum += xdatIn[datIndex - 1];
		sum += xdatIn[datIndex + 1];
		sum += xdatIn[datIndex - SIZE_X];
		sum += xdatIn[datIndex + SIZE_X];
		sum += xdatIn[datIndex - SIZE_X - 1];
		sum += xdatIn[datIndex + SIZE_X - 1];
		sum += xdatIn[datIndex - SIZE_X + 1];
		sum += xdatIn[datIndex + SIZE_X + 1];
		sum *= beta;
		
		xdatOut[datIndex] = sum;
		
		datIndex++;
	}
    
}
