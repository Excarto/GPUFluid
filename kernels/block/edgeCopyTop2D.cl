
// Copy values from just inside the top and bottom edges onto the top and bottom edges,
// possibly with a sign change

__kernel
void edgeCopyTop2D(__global float* dat){
	
	uint id = get_global_id(0);
	
	uint datShift = SIZE_X*(SIZE_Y - 1);
	
    dat[id] = MULT*dat[id + SIZE_X];
	dat[id + datShift] = MULT*dat[id + datShift - SIZE_X];
}
