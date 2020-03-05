import org.jocl.*;

public interface CLOperation{
	abstract public cl_event[] enqueue(cl_event[] inEvents);
}
