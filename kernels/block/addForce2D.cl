
// Impart an acceleration due to a fixed force field

__kernel
void addForce2D(__global const float* forcedat,
		  __global float* veldat){
	
	uint id = get_global_id(0);
	
	veldat[id] += DELTA*forcedat[id];
}
