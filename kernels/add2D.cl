
__kernel
void add2D(__global const float* indat,
		  __global float* outdat){
	
	uint id = get_global_id(0);
	
	outdat[id] += indat[id];
}
