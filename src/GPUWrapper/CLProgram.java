import static org.jocl.CL.*;
import java.io.*;
import org.jocl.*;
import java.util.*;

public class CLProgram{
	
	public final String name;
    public final CLDevice device;
    public final long[] nThreads, groupSize;
    public cl_program program;
    
    private final StringBuilder builder;
    private final ArrayList<CLKernel> kernels;
    
	public CLProgram(CLDevice device, String name, long[] nThreads, long[] groupSize){
		this.device = device;
		this.name = name;
		this.nThreads = nThreads;
		this.groupSize = groupSize;
		
		kernels = new ArrayList<CLKernel>();
		
		builder = new StringBuilder();
		try{
			BufferedReader reader = new BufferedReader(new FileReader("kernels/" + name + ".cl"));
			String line;
			while ((line = reader.readLine()) != null)
				builder.append(line + "\n");
			reader.close();
		}catch (IOException e){
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	public void addConstants(String[][] constants){
		for (String[] pair : constants)
			builder.insert(0, "#define " + pair[0] + " " + pair[1] + "\n");
	}
	public void addConstant(String name, String value){
		builder.insert(0, "#define " + name + " " + value + "\n");
	}
	public void addConstant(String name, int value){
		builder.insert(0, "#define " + name + " " + value + "\n");
	}
	public void addConstant(String name, long value){
		builder.insert(0, "#define " + name + " " + value + "l\n");
	}
	public void addConstant(String name, double value){
		builder.insert(0, "#define " + name + " " + value + "f\n");
	}
	
	public CLKernel getKernel(){
		if (program == null)
			initialize();
		CLKernel kernel = new CLKernel(this);
		kernels.add(kernel);
		return kernel;
	}
	
	public void initialize(){
		builder.insert(0, "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n");
		
		String source = builder.toString();
		
		int status[] = new int[1];
        program = clCreateProgramWithSource(
            device.context, 1, new String[]{source}, new long[]{source.length()}, status);
        String options;
        if (Fluid.DEBUG){
        	options = "-g";
        }else{
        	options = "-cl-fast-relaxed-math";//"-cl-uniform-work-group-size";
        }
        status[0] = clBuildProgram(program, 1, new cl_device_id[]{device.device}, options, null, null);
	}
	
	public void dispose(){
		for (CLKernel kernel : kernels)
			kernel.dispose();
		if (program != null)
			clReleaseProgram(program);
        program = null;
	}
}
