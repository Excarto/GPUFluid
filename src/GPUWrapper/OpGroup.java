import org.jocl.cl_event;
import java.util.*;

public class OpGroup implements CLOperation{
	
	private final CLOperation[] ops;
	private final ArrayList<cl_event> events;
	
	public OpGroup(CLOperation[] ops){
		this.ops = ops;
		events = new ArrayList<cl_event>();
	}
	
	public cl_event[] enqueue(cl_event[] inEvents){
		for (CLOperation op : ops){
			cl_event[] opOutEvents = op.enqueue(inEvents);
			for (cl_event event : opOutEvents)
				events.add(event);
		}
		
		cl_event[] outEvents = new cl_event[events.size()];
		for (int i = 0; i < events.size(); i++)
			outEvents[i] = events.get(i);
		events.clear();
		return outEvents;
	}
}
