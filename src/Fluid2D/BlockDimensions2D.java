import static java.lang.Math.*;

public class BlockDimensions2D{
	
	static final int DIV_POW = 21;
	static final int BLOCK_SIZE_DIVISOR = 4;
	
	public final int blockSize, cellsPerItem;
	public final int groupSize;
	public final int numBlocksX, numBlocksY;
	public final int sizeX, sizeY;
	public final int size;
	public final double score;
	
	public BlockDimensions2D(CLDevice device,
			int targetSizeX, int targetSizeY, int minSizeX, int minSizeY,
			int datPerCell, long maxMem, int maxLocalMem, boolean evenRows, int regionDivisor){
		int bytesPerCell = 4*datPerCell;
		
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
			
			int numFronts = iceil(numBlocksX*numBlocksY, device.numComputeUnits);
			double frontScore = pow(numBlocksX*numBlocksY/(double)(numFronts*device.numComputeUnits), 0.5);
			double blockScore = pow((ibs*ibs)/((double)(ibs+2)*(ibs+2)), 1.0);
			double sizeScore = pow(sizeX*sizeY/(double)(targetSizeX*targetSizeY), 0.25);
			
			int cellsPerBlock = ibs*ibs;
			int minCellsPer = iceil(cellsPerBlock, device.maxGroupSize);
			for (int icpi = minCellsPer; icpi <= cellsPerBlock; icpi++){
				if ((evenRows ? ibs : cellsPerBlock) % icpi != 0)
					continue;
				
				if (icpi != 1)
					continue;
					
				int groupSize = cellsPerBlock/icpi;
				double groupScore = groupSize/(double)device.maxGroupSize;
				
				double score = frontScore*blockScore*sizeScore*groupScore;
				if (score > optimScore){
					optimScore = score;
					optimBlockSize = ibs;
					optimCellsPerItem = icpi;
				}
			}
			
		}
		
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
	
	static int iceil(int num, int den){
		return num%den == 0 ? num/den : num/den+1;
	}
}
