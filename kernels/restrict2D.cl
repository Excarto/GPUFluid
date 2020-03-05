
__kernel
void restrict2D(__global const float* indat,
          __global float* outdat){
	
	uint blockX = get_group_id(0);
	uint blockY = get_group_id(1);
	uint localId = get_local_id(0);
	
	uint blockCellX = blockX*BLOCK_SIZE;
	uint blockCellY = blockY*BLOCK_SIZE;
	
	uint inBlockCellX = blockX*IN_BLOCK_SIZE;
	uint inBlockCellY = blockY*IN_BLOCK_SIZE;
	
	// OPERATE AND OUTPUT
	
	uint cell = localId*CELLS_PER_ITEM;
	uint cellY = (cell*ITEM_DIV_CONST) >> DIV_POW;
	uint cellX = cell - cellY*BLOCK_SIZE;
	cellX++;
	cellY++;
	
	uint inIndex = (inBlockCellY + 2*cellY - 1)*IN_SIZE_X + (inBlockCellX + 2*cellX - 1);
	uint outIndex = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float newVal = indat[inIndex]/4
				+ indat[inIndex - 1]/8
				+ indat[inIndex + 1]/8
				+ indat[inIndex - IN_SIZE_X]/8
				+ indat[inIndex + IN_SIZE_X]/8
				+ indat[inIndex - IN_SIZE_X - 1]/16
				+ indat[inIndex - IN_SIZE_X + 1]/16
				+ indat[inIndex + IN_SIZE_X - 1]/16
				+ indat[inIndex + IN_SIZE_X + 1]/16;
		outdat[outIndex] = newVal;
		
		outIndex++;
		inIndex += 2;
	}
    
}
