
// Restriction of a 2Nx2N grid to an NxN grid. Used as part of a multigrid solver.
// This is implemented as a block operation, with threads in the group synchronously loading the block data
// into a local buffer before operating to avoid redundant memory transfers

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
	
	// LOAD
	
	__local float block[IN_BLOCK_ROW_LEN*IN_BLOCK_ROW_LEN];
	
	uint blockShift = 0;
	uint indatShift = inBlockCellY*IN_SIZE_X + inBlockCellX;
#pragma unroll
	for (uint i = 0; i < IN_BLOCK_ROW_LEN; i++){
		async_work_group_copy(
				block + blockShift,
				indat + indatShift,
				IN_BLOCK_ROW_LEN,
				0);
		blockShift += IN_BLOCK_ROW_LEN;
		indatShift += IN_SIZE_X;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// OPERATE AND OUTPUT
	
	uint cell = localId*CELLS_PER_ITEM;
	uint cellY = (cell*ITEM_DIV_CONST) >> DIV_POW;
	uint cellX = cell - cellY*BLOCK_SIZE;
	cellX++;
	cellY++;
	
	uint blockIndex = (2*cellY - 1)*IN_BLOCK_ROW_LEN + (2*cellX - 1);
	uint outIndex = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float newVal = block[blockIndex]/4
				+ block[blockIndex - 1]/8
				+ block[blockIndex + 1]/8
				+ block[blockIndex - IN_BLOCK_ROW_LEN]/8
				+ block[blockIndex + IN_BLOCK_ROW_LEN]/8
				+ block[blockIndex - IN_BLOCK_ROW_LEN - 1]/16
				+ block[blockIndex - IN_BLOCK_ROW_LEN + 1]/16
				+ block[blockIndex + IN_BLOCK_ROW_LEN - 1]/16
				+ block[blockIndex + IN_BLOCK_ROW_LEN + 1]/16;
		outdat[outIndex] = newVal;
		
		outIndex++;
		blockIndex += 2;
	}
    
}
