import static org.jocl.CL.*;
import org.jocl.*;

public class CLKernel implements CLOperation{
	
    private final CLProgram program;
    private cl_kernel kernel;
    
	public CLKernel(CLProgram program){
		this.program = program;
		int status[] = new int[1];
		kernel = clCreateKernel(program.program, program.name, status);
	}
	
	public cl_event[] enqueue(cl_event[] inEvents){
		cl_event[] outEvent = new cl_event[]{program.device.getEvent()};
		clEnqueueNDRangeKernel(
	            program.device.getQueue(), kernel, program.nThreads.length, null,
	            program.nThreads, program.groupSize,
	            inEvents.length, inEvents.length == 0 ? null : inEvents, outEvent[0]);
		return outEvent;
	}
	
	public void setArg(int index, CLBuffer buff){
		clSetKernelArg(kernel, index, Sizeof.cl_mem, Pointer.to(buff.buffer));
	}
	
	public void setArg(int index, int val){
		clSetKernelArg(kernel, index, 4, Pointer.to(new int[]{val})); 
	}
	
	public void setArg(int index, float val){
		clSetKernelArg(kernel, index, 4, Pointer.to(new float[]{val})); 
	}
	
	public void dispose(){
		if (kernel != null)
			clReleaseKernel(kernel);
        kernel = null;
	}
}
