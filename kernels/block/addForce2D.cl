
__kernel
void addForce2D(__global const float* forcedat,
		  __global float* veldat){
	
	uint id = get_global_id(0);
	
	veldat[id] += DELTA*forcedat[id];
}
