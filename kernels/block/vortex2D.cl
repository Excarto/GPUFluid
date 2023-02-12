
// Correction term to add vorticity back into the velocity fields, as there is an artificial
// loss between iterations. Cannot currently find the reference for this term

__kernel
void vortex2D(__global const float* invelX,
		  __global const float* invelY,
          __global float* outvelX,
		  __global float* outvelY){
	
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
	uint index = (blockCellY + cellY)*SIZE_X + (blockCellX + cellX);
	
#pragma unroll
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		uint indexPY = index + SIZE_X;
		uint indexMY = index - SIZE_X;
		
		float omega = (invelY[index+1] - invelY[index-1] - invelX[indexPY] + invelX[indexMY])/2;
		
		float etaX = (invelY[index+1] + invelY[index-1] - 2*invelY[index])
				- (invelX[indexPY+1] + invelX[indexMY-1] - invelX[indexPY-1] - invelX[indexMY+1])/4;
		float etaY = (invelY[indexPY+1] + invelY[indexMY-1] - invelY[indexPY-1] - invelY[indexMY+1])/4
				- (invelX[indexPY] + invelX[indexMY] - 2*invelX[index]);
		float2 etaNorm = fast_normalize((float2)(etaX, etaY));
		
		float scale = fabs(omega)*EPSILON*DELTA;
		outvelX[index] = invelX[index] + scale*etaNorm.y;
		outvelY[index] = invelY[index] - scale*etaNorm.x;
		
		index++;
	}
    
}
