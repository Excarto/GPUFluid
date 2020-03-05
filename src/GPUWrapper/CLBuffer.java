import static java.lang.Math.*;
import static org.jocl.CL.*;
import org.jocl.*;

public class CLBuffer{
	
	final CLDevice device;
	final cl_mem buffer;
	final int size;
	final long nBytes;
	final long flags;
	final boolean isSubBuffer;
	
	public CLBuffer(CLDevice device, int size, boolean readable, boolean writable){
		this.device = device;
		this.nBytes = device.addrAlignBytes*lceil(4l*size, device.addrAlignBytes);
		this.size = (int)(this.nBytes/4);
		
		int status[] = new int[1];
		long flags = CL_MEM_READ_WRITE;
		if (!readable && !writable){
			flags = flags | CL_MEM_HOST_NO_ACCESS;
		}else if (!readable){
			flags = flags | CL_MEM_HOST_WRITE_ONLY;
		}else if (!writable){
			flags = flags | CL_MEM_HOST_READ_ONLY;
		}
		this.flags = flags;
		buffer = clCreateBuffer(device.context, flags, nBytes, null, status);
		isSubBuffer = false;
	}
	
	public CLBuffer(CLBuffer existing, int offsetSize, int size){
		this.device = existing.device;
		this.nBytes = device.addrAlignBytes*lceil(4l*size, device.addrAlignBytes);
		this.size = (int)(this.nBytes/4);
		this.flags = existing.flags;
		
		cl_buffer_region info = new cl_buffer_region();
		info.size = nBytes;
		info.origin = 4l*offsetSize;
		int status[] = new int[1];
		buffer = clCreateSubBuffer(existing.buffer, existing.flags, CL_BUFFER_CREATE_TYPE_REGION, info, status);
		isSubBuffer = true;
	}
	
	public void write(int[] data){
		long len = min(nBytes, 4l*data.length);
		clEnqueueWriteBuffer(
				device.getQueue(), buffer, CL_TRUE, 0, len, Pointer.to(data), 0, null, null);
	}
	
	public void write(float[] data){
		long len = min(nBytes, 4l*data.length);
		clEnqueueWriteBuffer(
				device.getQueue(), buffer, CL_TRUE, 0, len, Pointer.to(data), 0, null, null);
	}
	
	public void read(int[] data){
		long len = min(nBytes, 4l*data.length);
		clEnqueueReadBuffer(
				device.getQueue(), buffer, CL_TRUE, 0, len, Pointer.to(data), 0, null, null);
	}
	
	public void read(float[] data){
		long len = min(nBytes, 4l*data.length);
		clEnqueueReadBuffer(
				device.getQueue(), buffer, CL_TRUE, 0, len, Pointer.to(data), 0, null, null);
	}
	
	public void dispose(){
		clReleaseMemObject(buffer);
	}
	
	static long lceil(long num, long den){
		return num%den == 0 ? num/den : num/den+1;
	}
}
