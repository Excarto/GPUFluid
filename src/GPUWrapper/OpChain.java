import org.jocl.cl_event;

// An operation defined by a set of suboperations which are executed in sequence

public class OpChain implements CLOperation{
	
	private final CLOperation[] ops;
	
	public OpChain(CLOperation[] ops){
		this.ops = ops;
	}
	
	public cl_event[] enqueue(cl_event[] inEvents){
		cl_event[] events = inEvents;
		for (CLOperation op : ops)
			events = op.enqueue(events);
		return events;
	}
}
