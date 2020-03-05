
__kernel
void prolongate2D(__global const float* indat,
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
	uint inIndex = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	
	uint outCellX = 2*(blockCellX + cellX) - 1;
	uint outCellY = 2*(blockCellY + cellY) - 1;
	uint outIndex = outCellY*OUT_SIZE_X + outCellX;
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float val = indat[inIndex];
		float valRight = indat[inIndex+1];
		float valBot = indat[inIndex+SIZE_X];
		float valDiag = indat[inIndex+SIZE_X+1];
		// compute 4 vals
		outdat[outIndex] = val;
		outdat[outIndex + 1] = val/2 + valRight/2;
		outdat[outIndex + 2*SIZE_X] = val/2 + valBot/2;
		outdat[outIndex + 2*SIZE_X + 1] = val/4 + valRight/4 + valBot/4 + valDiag/4;
		
		outIndex += 2;
		inIndex++;
	}
    
}
