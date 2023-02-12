
// Advection of a quantity within a velocity field using simple linear extrapolation of trajectory.
// This is implemented as a block operation, with threads in the group synchronously loading the block data
// into a local buffer before operating to avoid redundant memory transfers

__kernel
void advect2D(__global const float* indat,
          __global float* outdat,
		  __global const float* vxdat,
		  __global const float* vydat){
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
	//uint endCellX = cellX + CELLS_PER_ITEM;
	uint datIndex = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	uint blockIndex = cellY*BLOCK_ROW_LEN + cellX;
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float dX = -DELTA*vxdat[datIndex];
		float dY = -DELTA*vydat[datIndex];
		
		uint blockIndexOffsetX = (int)sign(dX);
		uint blockDxIndex = blockIndex + blockIndexOffsetX;
		uint blockDyIndex = blockIndex + (int)sign(dY)*BLOCK_ROW_LEN;
		uint blockDxDyIndex = blockDyIndex + blockIndexOffsetX;
		
		dX = fabs(dX);
		dY = fabs(dY);
		
		float amt = (1.0f - dX)*(1.0f - dY);
		float amtDx = dX*(1.0f - dY);
		float amtDy = (1.0f - dX)*dY;
		float amtDxDy = dX*dY;
		
		float newVal = amt*block[blockIndex]
				+ amtDx*block[blockDxIndex]
				+ amtDy*block[blockDyIndex]
				+ amtDxDy*block[blockDxDyIndex];
		outdat[datIndex] = newVal;
		
		//cellX++;
		datIndex++;
		blockIndex++;
	}
    
}
