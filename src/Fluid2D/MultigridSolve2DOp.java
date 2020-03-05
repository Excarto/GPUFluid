import org.jocl.cl_event;

public class MultigridSolve2DOp implements CLOperation{
	private final CLOperation op;
	
	public MultigridSolve2DOp(CLDevice device,
			CLBuffer inputBuf, CLBuffer solutionBuf, CLBuffer workBuf,
			BlockDimensions2D dim, int[] numIterations,
			double alpha, double beta,
			boolean positiveBoundary){
		
		int numGrids = numIterations.length;
		
		if (numGrids == 0){
			op = new OpChain(new CLOperation[]{});
			return;
		}
		
		if (numGrids == 1){
			op = new JacobiSolve2DOp(device,
					inputBuf, solutionBuf,
					dim, numIterations[0],
					alpha, beta,
					positiveBoundary, true);
			return;
		}
		
		BlockDimensions2D[] gridDims = new BlockDimensions2D[numGrids];
		BlockDimensions2D[] restrictDims = new BlockDimensions2D[numGrids];
		String[][][] dimConstants = new String[numGrids][][];
		gridDims[0] = dim;
		dimConstants[0] = dim.getConstants();
		for (int i = 1; i < numGrids; i++){
			BlockDimensions2D nextStage = gridDims[i-1];
			int sizeX = (nextStage.sizeX - 2)/2 + 2;
			int sizeY = (nextStage.sizeY - 2)/2 + 2;
			gridDims[i] = new BlockDimensions2D(device,
				sizeX, sizeY, sizeX, sizeY,
				1, 4l*dim.size, device.localMemSize-1024, true, 4);
			restrictDims[i] = new BlockDimensions2D(device,
				sizeX, sizeY, sizeX, sizeY,
				1, 4l*dim.size, (device.localMemSize-1024)/4, true, 4);
			dimConstants[i] = gridDims[i].getConstants();
		}
		
		CLBuffer[] inputBufs = new CLBuffer[numGrids];
		inputBufs[0] = inputBuf;
		int bufOffset = 0;
		for (int i = 1; i < numGrids; i++){
			inputBufs[i] = device.getBuffer(solutionBuf, bufOffset, gridDims[i].size);
			bufOffset += inputBufs[i].size;
		}
		CLBuffer workBuf2 = device.getBuffer(solutionBuf, bufOffset, inputBufs[1].size);
		
		CLBuffer[] solutionBufs = new CLBuffer[numGrids];
		solutionBufs[0] = solutionBuf;
		solutionBufs[1] = workBuf;
		if (numGrids > 2)
			solutionBufs[2] = workBuf2;
		for (int i = 3; i < numGrids; i++)
			solutionBufs[i] = solutionBufs[i-2];
		
		CLProgram[][] edgePrograms = new CLProgram[numGrids][2];
		for (int i = 0; i < numGrids; i++){
			CLProgram copyTopProgram = device.getProgram("edgeCopyTop2D",
			new long[]{gridDims[i].sizeX},
			new long[]{gridDims[i].sizeX/(device.numComputeUnits - 1)});
			copyTopProgram.addConstants(dimConstants[i]);
			copyTopProgram.addConstant("MULT", positiveBoundary ? 1 : -1);
			
			CLProgram copySideProgram = device.getProgram("edgeCopySide2D",
					new long[]{gridDims[i].sizeY},
					new long[]{gridDims[i].sizeY/(device.numComputeUnits - 1)});
			copySideProgram.addConstants(dimConstants[i]);
			copySideProgram.addConstant("MULT", positiveBoundary ? 1 : -1);
			
			edgePrograms[i][0] = copyTopProgram;
			edgePrograms[i][1] = copySideProgram;
		}
		
		CLProgram[] restrictPrograms = new CLProgram[numGrids];
		for (int i = 0; i < numGrids-1; i++){
			BlockDimensions2D outDim = restrictDims[i+1];
			restrictPrograms[i] = device.getProgram("restrict2D",
					new long[]{outDim.groupSize*outDim.numBlocksX, outDim.numBlocksY},
					new long[]{outDim.groupSize, 1});
			restrictPrograms[i].addConstants(outDim.getConstants());
			restrictPrograms[i].addConstant("IN_BLOCK_SIZE", outDim.blockSize*2);
			restrictPrograms[i].addConstant("IN_BLOCK_ROW_LEN", outDim.blockSize*2 + 2);
			restrictPrograms[i].addConstant("IN_SIZE_X", (outDim.sizeX - 2)*2 + 2);
		}
		
		CLProgram[] prolongatePrograms = new CLProgram[numGrids];
		for (int i = 1; i < numGrids; i++){
			BlockDimensions2D inDim = gridDims[i];
			String[][] inConstants = dimConstants[i];
			BlockDimensions2D outDim = gridDims[i-1];
			prolongatePrograms[i] = device.getProgram("prolongate2D",
					new long[]{inDim.groupSize*inDim.numBlocksX, inDim.numBlocksY},
					new long[]{inDim.groupSize, 1});
			prolongatePrograms[i].addConstants(inConstants);
			prolongatePrograms[i].addConstant("OUT_SIZE_X", outDim.sizeX);
		}
		
		CLOperation[] restrictOps = new CLOperation[numGrids];
		for (int i = 0; i < numGrids-1; i++){
			CLKernel restrict = restrictPrograms[i].getKernel();
			restrict.setArg(0, inputBufs[i]);
			restrict.setArg(1, inputBufs[i+1]);
			restrictOps[i] = new OpChain(new CLOperation[]{
				restrict,
				getEdgeOp(edgePrograms[i+1], inputBufs[i+1])
			});
		}
		
		CLOperation[] solveOps = new CLOperation[numGrids];
		CLOperation[] prolongateOps = new CLOperation[numGrids];
		for (int i = numGrids-1; i >= 0; i--){
			solveOps[i] = new JacobiSolve2DOp(device,
					inputBufs[i], solutionBufs[i],
					gridDims[i], numIterations[i],
					alpha, beta,
					positiveBoundary, i == numGrids-1);
			
			if (i > 0){
				CLKernel prolongate = prolongatePrograms[i].getKernel();
				prolongate.setArg(0, solutionBufs[i]);
				prolongate.setArg(1, solutionBufs[i-1]);
				prolongateOps[i] = new OpChain(new CLOperation[]{
					prolongate,
					getEdgeOp(edgePrograms[i-1], solutionBufs[i-1])
				});
			}
		}
		
		CLOperation[] ops = new CLOperation[numGrids + 2*(numGrids-1)];
		for (int i = 0; i < numGrids-1; i++)
			ops[i] = restrictOps[i];
		int index = numGrids-1;
		for (int i = numGrids-1; i >= 0; i--){
			ops[index] = solveOps[i];
			if (i > 0)
				ops[index+1] = prolongateOps[i];
			index += 2;
		}
		
		op = new OpChain(ops);
	}
	
	private static CLOperation getEdgeOp(CLProgram[] edgePrograms, CLBuffer buf){
		CLKernel edge0 = edgePrograms[0].getKernel();
		edge0.setArg(0, buf);
		CLKernel edge1 = edgePrograms[1].getKernel();
		edge1.setArg(0, buf);
		return new OpGroup(new CLOperation[]{edge0, edge1});
	}
	
	public cl_event[] enqueue(cl_event[] inEvents){
		return op.enqueue(inEvents);
	}
}
