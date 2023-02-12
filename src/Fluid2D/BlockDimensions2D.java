import static java.lang.Math.*;

// Represents square computational block of full 2D data array. Needed for kernels that operate on spacially nearby elements,
// i.e. most physical operations. The size of the block is chosen to optimize resource use, dependent on the device.
// The boundary cells are assumed to not be included in the region covered by blocks, so that the total dimensions
// are given by (numBlocksX*blockSize + 2) by (numBlocksY*blockSize + 2)

public class BlockDimensions2D{
	
	static final int DIV_POW = 21; // Used for fast integer division
	static final int BLOCK_SIZE_DIVISOR = 4;
	
	public final int blockSize; /// Number of elements in each dimension of the block (block is square)
	public final int cellsPerItem; // Number of block elements processed by each OpenCL thread
	public final int groupSize; // OpenCL group size for use with this block, i.e. the number of threads used to process a block
	public final int numBlocksX, numBlocksY; // Full data array is split into this number of blocks in each dimension
	public final int sizeX, sizeY; // Numbers of elements in each dimension of the full data array
	public final int size; // Total number of elements in the full data array
	public final double score;
	
	public BlockDimensions2D(CLDevice device,
			int targetSizeX, int targetSizeY, int minSizeX, int minSizeY,
			int datPerCell, long maxMem, int maxLocalMem, boolean evenRows, int regionDivisor){
		int bytesPerCell = 4*datPerCell;
		
		// Block size optimization loop
		int optimBlockSize = 1;
		int optimCellsPerItem = 1;
		double optimScore = 0.0;
		for (int ibs = 1; ; ibs++){
			
			int localMemUse = bytesPerCell*(ibs+2)*(ibs+2);
			if (localMemUse > maxLocalMem)
				break;
			
			int numBlocksX = (targetSizeX - 2)/ibs;
			int numBlocksY = (targetSizeY - 2)/ibs;
			int sizeX = 2 + numBlocksX*ibs;
			int sizeY = 2 + numBlocksY*ibs;
			
			if (sizeX < minSizeX || sizeY < minSizeY)
				continue;
			
			if (ibs%BLOCK_SIZE_DIVISOR != 0)
				continue;
			
			if ((numBlocksX*ibs)%regionDivisor != 0 || (numBlocksY*ibs)%regionDivisor != 0)
				continue;
			
			long memUse = bytesPerCell*(long)sizeX*sizeY;
			if (memUse > maxMem)
				continue;
			
			// This is a valid block size satisfying the constraints. Compute some hueristic score values
			// to approximate performance with this block size
			int numFronts = iceil(numBlocksX*numBlocksY, device.numComputeUnits);
			double frontScore = pow(numBlocksX*numBlocksY/(double)(numFronts*device.numComputeUnits), 0.5);
			double blockScore = pow((ibs*ibs)/((double)(ibs+2)*(ibs+2)), 1.0);
			double sizeScore = pow(sizeX*sizeY/(double)(targetSizeX*targetSizeY), 0.25);
			
			// Number of sells per thread optimization loop
			int cellsPerBlock = ibs*ibs;
			int minCellsPer = iceil(cellsPerBlock, device.maxGroupSize);
			for (int icpi = minCellsPer; icpi <= cellsPerBlock; icpi++){
				if ((evenRows ? ibs : cellsPerBlock) % icpi != 0)
					continue;
				
				if (icpi != 1)
					continue;
				
				int groupSize = cellsPerBlock/icpi;
				double groupScore = groupSize/(double)device.maxGroupSize;
				
				// Compute total score for this (block size, cells per item) pair
				double score = frontScore*blockScore*sizeScore*groupScore;
				if (score > optimScore){
					optimScore = score;
					optimBlockSize = ibs;
					optimCellsPerItem = icpi;
				}
			}
			
		} // End block size optimization loop
		
		this.blockSize = optimBlockSize;
		this.cellsPerItem = optimCellsPerItem;
		this.numBlocksX = (targetSizeX - 2)/blockSize;
		this.numBlocksY = (targetSizeY - 2)/blockSize;
		this.sizeX = 2 + numBlocksX*blockSize;
		this.sizeY = 2 + numBlocksY*blockSize;
		this.groupSize = blockSize*blockSize/cellsPerItem;
		this.size = sizeX*sizeY*datPerCell;
		this.score = optimScore;
		
		if (optimScore <= 0){
			System.out.println("Could not find block dimensions");
			System.exit(1);
		}
		System.out.println("(sizeX,sizeY,nblocksX,nblockY) = (" + sizeX + "," + sizeY + "," + numBlocksX + "," + numBlocksY + ")");
		System.out.println("(blockSize,cellsPerItem,groupSize) = (" + blockSize + "," + cellsPerItem + "," + groupSize + ")");
	}
	
	// Generate String array of associated constants, for use in compiling program
	public String[][] getConstants(){
		return new String[][]{
				{"SIZE_X", String.valueOf(sizeX)},
				{"SIZE_Y", String.valueOf(sizeY)},
				{"NBLOCKS_X", String.valueOf(numBlocksX)},
				{"NBLOCKS_Y", String.valueOf(numBlocksY)},
				{"GROUP_SIZE", String.valueOf(groupSize)},
				{"BLOCK_SIZE", String.valueOf(blockSize)},
				{"BLOCK_ROW_LEN", String.valueOf(blockSize + 2)},
				{"CELLS_PER_ITEM", String.valueOf(cellsPerItem)},
				{"DIV_POW", String.valueOf(DIV_POW)},
				{"ITEM_DIV_CONST", String.valueOf((1 << DIV_POW)/blockSize + 1)}
		};
	}
	
	// Ceil of division of integers
	static int iceil(int num, int den){
		return num%den == 0 ? num/den : num/den+1;
	}
}
