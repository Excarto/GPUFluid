
__kernel
void subGrad2D(__global const float* indat,
		  __global float* outdatX,
          __global float* outdatY){
	
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
		
		float diff = indat[datIndex + 1] - indat[datIndex - 1];
		outdatX[datIndex] -= 0.5f*diff;
		diff = indat[datIndex + SIZE_X] - indat[datIndex - SIZE_X];
		outdatY[datIndex] -= 0.5f*diff;
		
		datIndex++;
	}
    
}
