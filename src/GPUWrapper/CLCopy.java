import static org.jocl.CL.*;
import org.jocl.*;

// An operation for copying memory between buffers on the device. 

public class CLCopy implements CLOperation{
	
	private final CLDevice device;
	private final CLBuffer source, dest;
	private final long sourceOffset, destOffset, nBytes;
	
	public CLCopy(CLDevice device,
			CLBuffer source, long sourceOffset, CLBuffer dest, long destOffset, long len){
		this.device = device;
		this.source = source;
		this.dest = dest;
		this.sourceOffset = sourceOffset;
		this.destOffset = destOffset;
		this.nBytes = 4*len;
	}
	
	public cl_event[] enqueue(cl_event[] inEvents){
		cl_event[] outEvent = new cl_event[]{device.getEvent()};
		clEnqueueCopyBuffer(device.getQueue(),
				source.buffer, dest.buffer, sourceOffset, destOffset, nBytes,
				inEvents.length, inEvents.length == 0 ? null : inEvents, outEvent[0]);
		return outEvent;
	}

}
