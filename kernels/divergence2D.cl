
__kernel
void divergence2D(__global const float* indatX,
		  __global const float* indatY,
          __global float* outdat){
	
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
		
		float sum = indatX[datIndex + 1] - indatX[datIndex - 1];
		sum += indatY[datIndex + SIZE_X] - indatY[datIndex - SIZE_X];
		
		outdat[datIndex] = 0.5f*sum;
		
		datIndex++;
	}
    
}
