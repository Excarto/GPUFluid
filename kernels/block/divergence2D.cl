
// Compute divergence.
// This is implemented as a block operation, with threads in the group synchronously loading the block data
// into a local buffer before operating to avoid redundant memory transfers

__kernel
void divergence2D(__global const float* indatX,
		  __global const float* indatY,
          __global float* outdat){
		  //__global uint* debug){
	
	uint blockX = get_group_id(0);
	uint blockY = get_group_id(1);
	uint localId = get_local_id(0);
	
	uint blockCellX = blockX*BLOCK_SIZE;
	uint blockCellY = blockY*BLOCK_SIZE;
	
	// LOAD
	
	__local float block[BLOCK_ROW_LEN*BLOCK_ROW_LEN];
	
	uint blockShift = 0;
	uint indatShift = blockCellY*SIZE_X + blockCellX;
#pragma unroll
	for (uint i = 0; i < BLOCK_ROW_LEN; i++){
		async_work_group_copy(
				block + blockShift,
				indatX + indatShift,
				BLOCK_ROW_LEN,
				0);
		blockShift += BLOCK_ROW_LEN;
		indatShift += SIZE_X;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// OPERATE AND OUTPUT
	
	uint cell = localId*CELLS_PER_ITEM;
	uint cellY = (cell*ITEM_DIV_CONST) >> DIV_POW;
	uint cellX = cell - cellY*BLOCK_SIZE;
	cellX++;
	cellY++;
	//uint endCellX = cellX + CELLS_PER_ITEM;
	uint datIndex = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	uint blockIndex = cellY*BLOCK_ROW_LEN + cellX;
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float sum = block[blockIndex + 1] - block[blockIndex - 1];
		sum += indatY[datIndex + SIZE_X] - indatY[datIndex - SIZE_X];
		
		outdat[datIndex] = 0.5f*sum;
		
		//cellX++;
		datIndex++;
		blockIndex++;
	}
    
}
