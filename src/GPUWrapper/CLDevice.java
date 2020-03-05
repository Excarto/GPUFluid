import static org.jocl.CL.*;
import java.nio.*;
import org.jocl.*;
import java.util.*;

public class CLDevice{
	
	static final boolean ENABLE_TIMING = false;
	
	public final cl_device_id device;
	public final cl_context context;
	
	
	public final int numComputeUnits;
	public final int maxGroupSize;
	public final long globalMemSize;
	public final int localMemSize;
	public final boolean byteStore;
	public final int vectWidthFloat, vectWidthInt;
	public final int numDimensions;
	public final int numQueues;
	public final int addrAlignBytes;
	
	private final cl_command_queue[] queues;
	private int queueIndex;
	private ArrayList<CLProgram> programs;
	private ArrayList<CLBuffer> buffers;
	private ArrayList<cl_event> events;
	
	public CLDevice(boolean parallel){
		int temp[] = new int[1];
		
		programs = new ArrayList<CLProgram>();
		buffers  = new ArrayList<CLBuffer>();
		events = new ArrayList<cl_event>();
		
        clGetPlatformIDs(0, null, temp);
        int numPlatforms = temp[0];
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        
        cl_platform_id devicePlatform = null;
        cl_device_id deviceId = null;
        for (cl_platform_id platformId : platforms){
        	
            clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, null, temp);
            int numDevices = temp[0];
            cl_device_id platformDevices[] = new cl_device_id[numDevices];
            clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, numDevices, platformDevices, null);
            
            for (cl_device_id platformDevice : platformDevices){
            	long deviceType = getLong(platformDevice, CL_DEVICE_TYPE);
            	if ((deviceType & CL_DEVICE_TYPE_GPU) != 0){
            		deviceId = platformDevice;
            		devicePlatform = platformId;
            	}
            }
        }
        
        device = deviceId;
        if (device == null){
        	System.out.println("ERROR: Unable to find GPU");
        	System.exit(1);
        }
        
        numComputeUnits = getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
        maxGroupSize = (int)getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        localMemSize = (int)getLong(device, CL_DEVICE_LOCAL_MEM_SIZE);
        globalMemSize = getLong(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        vectWidthFloat = getInt(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
        vectWidthInt = getInt(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
        numDimensions = getInt(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        numQueues = parallel ? getInt(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES) : 1;
        addrAlignBytes = getInt(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN)/8;
        
        byte deviceExtensions[] = new byte[2048];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 
            deviceExtensions.length, Pointer.to(deviceExtensions), null);
        String deviceExtensionsString = new String(deviceExtensions);
        byteStore = deviceExtensionsString.contains("cl_khr_byte_addressable_store");
        
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, devicePlatform);
		context = clCreateContext(
            contextProperties, 1, new cl_device_id[]{device}, null, null, null);
		
		cl_queue_properties properties = new cl_queue_properties();
		if (ENABLE_TIMING)
            properties.addProperty(CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE);
		//properties.addProperty(CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		queues = new cl_command_queue[numQueues];
		for (int i = 0; i < numQueues; i++)
			queues[i] = clCreateCommandQueueWithProperties(context, device, properties, null);
	}
	
	public CLProgram getProgram(String name, long[] nThreads, long[] groupSize){
		CLProgram program = new CLProgram(this, name, nThreads, groupSize);
		programs.add(program);
		return program;
	}
	
	public CLBuffer getBuffer(int size, boolean readable, boolean writable){
		CLBuffer buffer = new CLBuffer(this, size, readable, writable);
		buffers.add(buffer);
		return buffer;
	}
	
	public CLBuffer getBuffer(CLBuffer existing, int offsetSize, int size){
		CLBuffer buffer = new CLBuffer(existing, offsetSize, size);
		buffers.add(buffer);
		return buffer;
	}
	
	public cl_command_queue getQueue(){
		return queues[(queueIndex++)%numQueues];
	}
	
	public cl_event getEvent(){
		cl_event event = new cl_event();
		events.add(event);
		return event;
	}
	
	public void run(CLOperation op){
		cl_event[] finalEvents = op.enqueue(new cl_event[]{});
		clWaitForEvents(finalEvents.length, finalEvents);
		
		while (!events.isEmpty())
			clReleaseEvent(events.remove(events.size()-1));
	}
	
	public int getTotalMB(){
		int sum = 0;
		for (CLBuffer buffer : buffers){
			if (!buffer.isSubBuffer)
				sum += buffer.nBytes/(1024*1024);
		}
		return sum;
	}
	
	public void dispose(){
    	if (programs == null)
    		return;
    	for (CLProgram program : programs)
    		program.dispose();
    	for (CLBuffer buffer : buffers)
    		buffer.dispose();
    	for (cl_command_queue queue : queues)
    		clReleaseCommandQueue(queue);
    	clReleaseContext(context);
    	programs = null;
    	buffers = null;
    }
	
	private int getInt(cl_device_id device, int param){
        int values[] = new int[1];
        clGetDeviceInfo(device, param, Sizeof.cl_int, Pointer.to(values), null);
        return values[0];
    }
    
    private long getLong(cl_device_id device, int param){
        long values[] = new long[1];
        clGetDeviceInfo(device, param, Sizeof.cl_long, Pointer.to(values), null);
        return values[0];
    }
    
    private long getSize(cl_device_id device, int param){
        ByteBuffer buffer = ByteBuffer.allocate(Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, param, Sizeof.size_t, Pointer.to(buffer), null);
        return Sizeof.size_t == 4 ? buffer.getInt(0) : buffer.getLong(0);
    }
}
