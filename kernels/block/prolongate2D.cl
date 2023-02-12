
// Prolongation of an NxN grid to a 2Nx2N grid. Used as part of a multigrid solver.
// This is implemented as a block operation, with threads in the group synchronously loading the block data
// into a local buffer before operating to avoid redundant memory transfers

__kernel
void prolongate2D(__global const float* indat,
          __global float* outdat){
	
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
				indat + indatShift,
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
	uint blockIndex = cellY*BLOCK_ROW_LEN + cellX;
	
	uint outCellX = 2*(blockCellX + cellX) - 1;
	uint outCellY = 2*(blockCellY + cellY) - 1;
	uint outIndex = outCellY*OUT_SIZE_X + outCellX;
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float val = block[blockIndex];
		float valRight = block[blockIndex+1];
		float valBot = block[blockIndex+BLOCK_ROW_LEN];
		float valDiag = block[blockIndex+BLOCK_ROW_LEN+1];
		// compute 4 vals
		outdat[outIndex] = val;
		outdat[outIndex + 1] = val/2 + valRight/2;
		outdat[outIndex + 2*SIZE_X] = val/2 + valBot/2;
		outdat[outIndex + 2*SIZE_X + 1] = val/4 + valRight/4 + valBot/4 + valDiag/4;
		
		outIndex += 2;
		blockIndex++;
	}
    
}
