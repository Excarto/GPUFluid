import org.jocl.cl_event;

// Operation for satisfying boundary conditions by copying the data from the cells adjacent to the boundary
// to the cells on the boundary, possibly with a change of sign

public class EdgeCopyOp implements CLOperation{
	CLOperation op;
	
	public EdgeCopyOp(CLDevice device,
			CLBuffer buf,
			BlockDimensions2D dim,
			boolean positiveBoundary){
		
		CLProgram copyTopProgram = device.getProgram("edgeCopyTop2D",
				new long[]{dim.sizeX},
				null);//new long[]{dim.sizeX/(device.numComputeUnits - 1)});
		copyTopProgram.addConstants(dim.getConstants());
		copyTopProgram.addConstant("MULT", positiveBoundary ? 1 : -1);
		
		CLProgram copySideProgram = device.getProgram("edgeCopySide2D",
				new long[]{dim.sizeY},
				null);//new long[]{dim.sizeY/(device.numComputeUnits - 1)});
		copySideProgram.addConstants(dim.getConstants());
		copySideProgram.addConstant("MULT", positiveBoundary ? 1 : -1);
		
		CLKernel topEdgeOp = copyTopProgram.getKernel();
		topEdgeOp.setArg(0, buf);
		CLKernel sideEdgeOp = copySideProgram.getKernel();
		sideEdgeOp.setArg(0, buf);
		
		op = new OpChain(new CLOperation[]{topEdgeOp, sideEdgeOp});
	}
	
	public cl_event[] enqueue(cl_event[] inEvents){
		return op.enqueue(inEvents);
	}
}
