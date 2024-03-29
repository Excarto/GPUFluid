
// Copy values from just inside the left and right edges onto the left and right edges,
// possibly with a sign change

__kernel
void edgeCopySide2D(__global float* dat){
	
	uint id = get_global_id(0);
	
	uint datShiftLeft = SIZE_X*id;
	uint datShiftRight = datShiftLeft + SIZE_X - 1;
	
    dat[datShiftLeft] = MULT*dat[datShiftLeft + 1];
	dat[datShiftRight] = MULT*dat[datShiftRight - 1];
}
