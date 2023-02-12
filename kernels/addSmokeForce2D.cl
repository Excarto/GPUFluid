
// Add acceleration due to bouyancy force

__kernel
void addSmokeForce2D(__global const float* dyedat,
		__global const float* tempdat,
		__global float* velydat){
	
	uint id = get_global_id(0);
	
	float force = -dyedat[id]*SMOKE_WEIGHT;
	force += tempdat[id]*BUOYANCY;
	
	velydat[id] += DELTA*force;
}
