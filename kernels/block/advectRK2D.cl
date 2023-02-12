
// Advection of 3 quantities within a velocity field using 4th order Runge-Kutta for extrapolation of trajectory.
// This is implemented as a block operation, with threads in the group synchronously loading the block data
// into a local buffer before operating to avoid redundant memory transfers

struct BilinDat{
	float amt;
	float amtDx;
	float amtDy;
	float amtDxDy;
	uint index;
	uint indexDx;
	uint indexDy;
	uint indexDxDy;
};

void bilinDat(uint datIndex,
		float dX, float dY,
		struct BilinDat* bilin){
	
	int iX = (int)dX;
	int iY = (int)dY;
	
	float ratioX = fabs(dX - (float)iX);
	float ratioY = fabs(dY - (float)iY);
	bilin->amt = (1.0f - ratioX)*(1.0f - ratioY);
	bilin->amtDx = ratioX*(1.0f - ratioY);
	bilin->amtDy = (1.0f - ratioX)*ratioY;
	bilin->amtDxDy = ratioX*ratioY;
	
	int signX = (int)sign(dX);
	int signY = (int)sign(dY);
	uint index = datIndex + iY*SIZE_X + iX;
	bilin->index = index;
	bilin->indexDx = index + signX;
	bilin->indexDy = index + signY*SIZE_X;
	bilin->indexDxDy = index + signY*SIZE_X + signX;
}

float bilinInterp(__global const float* indat, struct BilinDat* bilin){
	return bilin->amt*indat[bilin->index]
			+ bilin->amtDx*indat[bilin->indexDx]
			+ bilin->amtDy*indat[bilin->indexDy]
			+ bilin->amtDxDy*indat[bilin->indexDxDy];
}

// 4th order Runge-Kutta
void RK4Step(__global const float* fxdat,
		__global const float* fydat,
		uint datIndex, float delta,
		float* pathX, float* pathY){
	
	float dX = delta*fxdat[datIndex];
	float dY = delta*fydat[datIndex];
	
	float sumX, sumY;
	float fX, fY;
	struct BilinDat bilin;
	
	sumX = dX;
	sumY = dY;
	
	bilinDat(datIndex, dX/2, dY/2, &bilin);
	fX = bilinInterp(fxdat, &bilin);
	fY = bilinInterp(fydat, &bilin);
	dX = delta*fX;
	dY = delta*fY;
	sumX += 2*dX;
	sumY += 2*dY;
	
	bilinDat(datIndex, dX/2, dY/2, &bilin);
	fX = bilinInterp(fxdat, &bilin);
	fY = bilinInterp(fydat, &bilin);
	dX = delta*fX;
	dY = delta*fY;
	sumX += 2*dX;
	sumY += 2*dY;
	
	bilinDat(datIndex, dX, dY, &bilin);
	fX = bilinInterp(fxdat, &bilin);
	fY = bilinInterp(fydat, &bilin);
	dX = delta*fX;
	dY = delta*fY;
	sumX += dX;
	sumY += dY;
	
	*pathX = sumX/6.0f;
	*pathY = sumY/6.0f;
}

__kernel
void advectRK2D(__global const float* vxdat, __global const float* vydat,
		  __global const float* indat1, __global float* outdat1,
		  __global const float* indat2, __global float* outdat2,
		  __global const float* indat3, __global float* outdat3){
	
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
	
	for (uint i = 0; i < CELLS_PER_ITEM; i++){
		
		float pathX, pathY;
		RK4Step(vxdat, vydat,
				datIndex, -DELTA,
				&pathX, &pathY);
		//pathX = -DELTA*vxdat[datIndex];
		//pathY = -DELTA*vydat[datIndex];
		
		struct BilinDat bilin;
		bilinDat(datIndex, pathX, pathY, &bilin);
		
		outdat1[datIndex] = bilinInterp(indat1, &bilin);
		outdat2[datIndex] = bilinInterp(indat2, &bilin);
		outdat3[datIndex] = bilinInterp(indat3, &bilin);
		
		datIndex++;
	}
    
}
