
public class Fluid2D{
	
	enum Stencil{
		JACOBI_5POINT("jacobi5P2D"),
		JACOBI_9POINT("jacobi9P2D");
		
		public final String name;
		Stencil(String name){
			this.name = name;
		}
	};
	
	static final Stencil STENCIL_TO_USE = Stencil.JACOBI_9POINT;
	
	static final int DATA_PER_CELL = 3;
	static final int VX_INDEX = 0;
	static final int VY_INDEX = 1;
	static final int DYE_INDEX = 2;
	static final int TEMP_INDEX = 3;
	static final int DIV_INDEX = 0;
	static final int P_INDEX = 1;
	static final int WORK_INDEX = 2;
	
	static final long SPARE_MEM = 128*1024*1042;
	
	static final int[] PRESSURE_SOLVE_GRID = new int[]{10, 30, 150};
	//static final int[] DIFFUSION_SOLVE_GRID = new int[]{10, 30, 150};
	
	float[] data;
	
	CLDevice device;
	CLBuffer[] dataBufs;
	CLBuffer[] forceBufs;
	CLBuffer sourceBuf;
	
	CLOperation iteration;
	BlockDimensions2D dim;
	double deltaTime;
	double viscosity;
	double vortexEps;
	
	public Fluid2D(int minSizeX, int minSizeY, int maxSizeX, int maxSizeY,
			double deltaTime, double viscosity, double vortexEps){
		this.deltaTime = deltaTime;
		this.viscosity = viscosity;
		this.vortexEps = vortexEps;
		
		device = new CLDevice(true);
		System.out.println("nqueues: " + device.numQueues);
		
		long maxMem = (device.globalMemSize - SPARE_MEM)/(1 + DATA_PER_CELL);
		dim = new BlockDimensions2D(device,
				maxSizeX, maxSizeY, minSizeX, minSizeY,
				1, maxMem, device.localMemSize-1024, true, 1 << (PRESSURE_SOLVE_GRID.length-1));
		data = new float[dim.sizeX*dim.sizeY];
		String[][] dimConstants = dim.getConstants();
		
		dataBufs = new CLBuffer[DATA_PER_CELL];
		CLBuffer[] workBufs = new CLBuffer[DATA_PER_CELL];
		
		forceBufs = new CLBuffer[2];
		for (int i = 0; i < forceBufs.length; i++)
			forceBufs[i] = device.getBuffer(data.length, false, true);
		
		sourceBuf = device.getBuffer(data.length, false, true);
		
		CLProgram advectProgram = device.getProgram("advect" + DATA_PER_CELL + "RK2D",
				new long[]{dim.groupSize*dim.numBlocksX, dim.numBlocksY},
				new long[]{dim.groupSize, 1});
		advectProgram.addConstants(dimConstants);
		advectProgram.addConstant("DELTA", deltaTime);
		
		CLProgram vortexProgram = device.getProgram("vortex2D",
				new long[]{dim.groupSize*dim.numBlocksX, dim.numBlocksY},
				new long[]{dim.groupSize, 1});
		vortexProgram.addConstants(dimConstants);
		vortexProgram.addConstant("DELTA", deltaTime);
		vortexProgram.addConstant("EPSILON", vortexEps);
		vortexProgram.addConstant("BORDER_SIZE", 2);
		
		CLProgram addProgram = device.getProgram("add2D",
				new long[]{dim.sizeX*dim.sizeY},
				new long[]{device.maxGroupSize});
		addProgram.addConstants(dimConstants);
		
		CLProgram smokeProgram = device.getProgram("addSmokeForce2D",
				new long[]{dim.sizeX*dim.sizeY},
				new long[]{device.maxGroupSize});
		smokeProgram.addConstant("SMOKE_WEIGHT", "0.1");
		smokeProgram.addConstant("BUOYANCY", "2.0");
		smokeProgram.addConstant("DELTA", deltaTime);
		
		CLProgram divProgram = device.getProgram("divergence2D",
				new long[]{dim.groupSize*dim.numBlocksX, dim.numBlocksY},
				new long[]{dim.groupSize, 1});
		divProgram.addConstants(dimConstants);
		
		CLProgram subGradProgram = device.getProgram("subGrad2D",
				new long[]{dim.groupSize*dim.numBlocksX, dim.numBlocksY},
				new long[]{dim.groupSize, 1});
		subGradProgram.addConstants(dimConstants);
		
		CLCopy[] copyOps = new CLCopy[DATA_PER_CELL];
		for (int i = 0; i < DATA_PER_CELL; i++){
			dataBufs[i] = device.getBuffer(data.length, true, true);
			workBufs[i] = device.getBuffer(data.length, true, false);
			copyOps[i] =  new CLCopy(device, dataBufs[i], 0, workBufs[i], 0, data.length);
		}
		
		CLKernel advectOp = advectProgram.getKernel();
		advectOp.setArg(0, workBufs[VX_INDEX]);
		advectOp.setArg(1, workBufs[VY_INDEX]);
		for (int i = 0; i < DATA_PER_CELL; i++){
			advectOp.setArg(2*i+2, workBufs[i]);
			advectOp.setArg(2*i+3, dataBufs[i]);
		}
		
		double diffuseAlpha = 0, diffuseBeta = 0;
		if (STENCIL_TO_USE == Stencil.JACOBI_5POINT){
			diffuseAlpha = 1/(viscosity*deltaTime);
			diffuseBeta = 1/(4.0 + diffuseAlpha);
		}else if (STENCIL_TO_USE == Stencil.JACOBI_9POINT){
			diffuseAlpha = 3/(viscosity*deltaTime);
			diffuseBeta = 1/(8.0 + diffuseAlpha);
		}
		
		CLOperation[] diffuseOps = new CLOperation[2];
		for (int i = VX_INDEX; i <= VY_INDEX; i++){
			/*diffuseOps[i] = new MultigridSolve2DOp(device,
					workBufs[i], dataBufs[i], workBufs[WORK_INDEX],
					dim, DIFFUSION_SOLVE_GRID,
					diffuseAlpha, diffuseBeta,
					false
			);*/
			diffuseOps[i] = new JacobiSolve2DOp(device,
					workBufs[i], dataBufs[i],
					dim, 20,
					diffuseAlpha, diffuseBeta,
					false, false
			);
		}
		if (viscosity == 0)
			diffuseOps = new CLOperation[0];
		
		CLKernel vortexOp = vortexProgram.getKernel();
		vortexOp.setArg(0, workBufs[VX_INDEX]);
		vortexOp.setArg(1, workBufs[VY_INDEX]);
		vortexOp.setArg(2, dataBufs[VX_INDEX]);
		vortexOp.setArg(3, dataBufs[VY_INDEX]);
		
		CLOperation[] forceOps = new CLOperation[2];
		for (int i = VX_INDEX; i <= VY_INDEX; i++){
			CLKernel ker = addProgram.getKernel();
			ker.setArg(0, forceBufs[i]);
			ker.setArg(1, dataBufs[i]);
			forceOps[i] = ker;
		}
		
		/*CLKernel dyeSourceOp = addProgram.getKernel();
		dyeSourceOp.setArg(0, sourceBuf);
		dyeSourceOp.setArg(1, dataBufs[DYE_INDEX]);
		CLKernel tempSourceOp = addProgram.getKernel();
		tempSourceOp.setArg(0, sourceBuf);
		tempSourceOp.setArg(1, dataBufs[TEMP_INDEX]);
		
		CLKernel smokeForceOp = smokeProgram.getKernel();
		smokeForceOp.setArg(0, dataBufs[DYE_INDEX]);
		smokeForceOp.setArg(1, dataBufs[TEMP_INDEX]);
		smokeForceOp.setArg(2, dataBufs[VY_INDEX]);*/
		
		CLOperation[] velEdgeOps = new CLOperation[2];
		for (int i = VX_INDEX; i <= VY_INDEX; i++)
			velEdgeOps[i] = new EdgeCopyOp(device, dataBufs[i], dim, false);
		
		CLKernel divOp = divProgram.getKernel();
		divOp.setArg(0, dataBufs[VX_INDEX]);
		divOp.setArg(1, dataBufs[VY_INDEX]);
		divOp.setArg(2, workBufs[DIV_INDEX]);
		
		double pressureAlpha = 0, pressureBeta = 0;
		if (STENCIL_TO_USE == Stencil.JACOBI_5POINT){
			pressureAlpha = -1;
			pressureBeta = 1/4.0;
		}else if (STENCIL_TO_USE == Stencil.JACOBI_9POINT){
			pressureAlpha = -3;
			pressureBeta = 1/8.0;
		}
		
		CLOperation pressureOp = new MultigridSolve2DOp(device,
				workBufs[DIV_INDEX], workBufs[P_INDEX], workBufs[WORK_INDEX],
				dim, PRESSURE_SOLVE_GRID,
				pressureAlpha, pressureBeta,
				true
		);
		/*GPUOperation pressureOp = new JacobiSolve2DOp(device,
				workBufs[DIV_INDEX], workBufs[P_INDEX],
				dim, 50,
				pressureAlpha, pressureBeta,
				true, true
		);*/
		
		CLKernel projectOp = subGradProgram.getKernel();
		projectOp.setArg(0, workBufs[P_INDEX]);
		projectOp.setArg(1, dataBufs[VX_INDEX]);
		projectOp.setArg(2, dataBufs[VY_INDEX]);
		
		iteration = new OpChain(new CLOperation[]{
				new OpGroup(copyOps),
				advectOp,
				new OpGroup(new CLOperation[]{copyOps[VX_INDEX], copyOps[VY_INDEX]}),
				new OpChain(diffuseOps),
				new OpGroup(new CLOperation[]{copyOps[VX_INDEX], copyOps[VY_INDEX]}),
				vortexOp,
				//new OpGroup(forceOps),
				//new OpGroup(new GPUOperation[]{dyeSourceOp, tempSourceOp}),
				//smokeForceOp,
				divOp,
				pressureOp,
				projectOp
		});
		
		System.out.println(device.getTotalMB() + " MB");
	}
	
	public void iterate(){
		
		device.run(iteration);
	}
	
	public void dispose(){
		device.dispose();
	}
	
	public void copyBack(int index){
		dataBufs[index].read(data);
	}
	
}
