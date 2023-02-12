import org.jocl.*;

// An operation that can be executed by the device. The inEvents argument gives a list of events that need
// to be completed before the operation will execute. The return value is an event that indicates when this
// operation is complete.

public interface CLOperation{
	abstract public cl_event[] enqueue(cl_event[] inEvents);
}
