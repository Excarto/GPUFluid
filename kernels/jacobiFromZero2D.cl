
// First iteration of a Jacobi solver, where the initial data is assumed to be zero.

__kernel
void jacobiFromZero2D(__global const float* bdat,
          __global float* xdat,
		  const float alpha){
		  //__global uint* debug){
	
	uint blockX = get_group_id(0);
	uint blockY = get_group_id(1);
	uint localId = get_local_id(0);
	
	uint blockCellX = blockX*BLOCK_SIZE;
	uint blockCellY = blockY*BLOCK_SIZE;
	
	// OPERATE AND OUTPUT
	
	uint cell = localId*CELLS_PER_ITEM;
	uint cellY = (cell*ITEM_DIV_CONST) >> DIV_POW;
	uint cellX = cell - cellY*BLOCK_SIZE;
	cellX++;
	cellY++;
	uint datIndex = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		float sum = alpha*bdat[datIndex];
		
		xdat[datIndex] = sum;
		
		datIndex++;
	}
    
}
