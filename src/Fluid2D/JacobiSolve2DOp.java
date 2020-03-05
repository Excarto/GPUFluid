import org.jocl.cl_event;

public class JacobiSolve2DOp implements CLOperation{
	CLOperation op;
	
	public JacobiSolve2DOp(CLDevice device,
			CLBuffer inputBuf, CLBuffer solutionBuf,
			BlockDimensions2D dim, int numIterations,
			double alpha, double beta,
			boolean positiveBoundary, boolean fromZero){
		
		String[][] constants = dim.getConstants();
		
		CLProgram zeroProgram = null;
		if (fromZero){
			zeroProgram = device.getProgram("jacobiFromZero2D",
			new long[]{dim.groupSize*dim.numBlocksX, dim.numBlocksY},
			new long[]{dim.groupSize, 1});
			zeroProgram.addConstants(constants);
		}
		
		CLProgram jacobiProgram = device.getProgram(Fluid2D.STENCIL_TO_USE.name,
			new long[]{dim.groupSize*dim.numBlocksX, dim.numBlocksY},
			new long[]{dim.groupSize, 1});
		jacobiProgram.addConstants(constants);
		
		CLKernel zeroOp = null;
		if (fromZero){
			zeroOp = zeroProgram.getKernel();
			zeroOp.setArg(0, inputBuf);
			zeroOp.setArg(1, solutionBuf);
			zeroOp.setArg(2, (float)alpha);
		}
		
		CLKernel jacobiOp = jacobiProgram.getKernel();
		jacobiOp.setArg(0, inputBuf);
		jacobiOp.setArg(1, solutionBuf);
		jacobiOp.setArg(2, solutionBuf);
		jacobiOp.setArg(3, (float)alpha);
		jacobiOp.setArg(4, (float)beta);
		
		CLOperation edgeOp = new EdgeCopyOp(device, solutionBuf, dim, positiveBoundary);
		
		CLOperation[] ops = new CLOperation[2*numIterations];
		for (int i = 0; i < numIterations; i++){
			ops[2*i + 0] = i == 0 && fromZero ? zeroOp : jacobiOp;
			ops[2*i + 1] = edgeOp;
		}
		op = new OpChain(ops);
		//op = new OpChain(new GPUOperation[]{zeroOp, edgeOp});
	}
	
	public cl_event[] enqueue(cl_event[] inEvents){
		return op.enqueue(inEvents);
	}
}
