import static java.lang.Math.*;
import org.jocl.*;

// Main class that creates a Fluid2D object, initializes it with some arbitrary starting data,
// and creates a FluidGUI object

public class Fluid{
	
	public static final boolean DEBUG = false;

	public static void main(String[] args){
		CL.setExceptionsEnabled(true);
		
		int minSizeX = 1000;//1920 + 2*FluidGUI.BORDER_SIZE - 200;
		int minSizeY = 1000;//1080 + 2*FluidGUI.BORDER_SIZE;
		int maxSizeX = minSizeX + 30;
		int maxSizeY = minSizeY + 30;
		
		double viscosity = 10.0;
		double deltaTime = 0.025;
		double vortexEps = (8.5 + random()*1.5)/pow(deltaTime, 0.25);
		
		Fluid2D fluid2D = new Fluid2D(minSizeX, minSizeY, maxSizeX, maxSizeY,
				deltaTime, viscosity, vortexEps);
		
		initializeData(fluid2D);
		sendForce(fluid2D);
		sendSource(fluid2D);
		
		try{
			Thread.sleep(500);
		}catch (Exception ex){}
		
		new FluidGUI(fluid2D);
		
	}
	
	static void initializeData(Fluid2D fluid){
		float[] data = fluid.data;
		
		zero(data);
		if (random() < 0.5){
			addBlob(data, fluid.dim, 0.85, 0.1, -1000.0*fluid.deltaTime, 0.20, false, false);
		}else
			addBlob(data, fluid.dim, 0.15, 0.1, 1000.0*fluid.deltaTime, 0.20, false, false);
		//addBlob(data, 0.3, 0.7,  1.0*deltaTime, 0.2);
		fluid.dataBufs[Fluid2D.VX_INDEX].write(data);
		zero(data);
		//addBlob(data, 0.4, 0.9, -1000.0*deltaTime, 0.20, false, false);
		//addBlob(data, 0.5, 0.9, -500.0*deltaTime, 0.05);
		fluid.dataBufs[Fluid2D.VY_INDEX].write(data);
		zero(data);
		//addBlob(data, 0.5, 0.5, 1.0, 0.6, false, false);
		addBlob(data, fluid.dim, 0.5, 0.5, 1.0, 0.8, false, false);
		fluid.dataBufs[Fluid2D.DYE_INDEX].write(data);
	}
	
	static void sendForce(Fluid2D fluid){
		float[] data = fluid.data;
		
		zero(data);
		//addBlob(data, 0.85, 0.10, -0.5*deltaTime, 0.1, false, false);
		fluid.forceBufs[Fluid2D.VX_INDEX].write(data);
		zero(data);
		fluid.forceBufs[Fluid2D.VY_INDEX].write(data);
	}
	
	static void sendSource(Fluid2D fluid){
		float[] data = fluid.data;
		
		zero(data);
		addBlob(data, fluid.dim, 0.5, 0.05, 0.1*fluid.deltaTime, 0.02, false, false);
		fluid.sourceBuf.write(data);
	}
	
	static void zero(float[] data){
		for (int i = 0; i < data.length; i++)
			data[i] = 0.0f;
	}
	
	static void addBlob(float[] data, BlockDimensions2D dim,
			double centerX, double centerY,
			double mag, double rad,
			boolean xComp, boolean yComp){
		centerX *= dim.sizeX;
		centerY *= dim.sizeY;
		rad *= min(dim.sizeX, dim.sizeY);
		
		for (int x = 0; x < dim.sizeX; x++){
			for (int y = 0; y < dim.sizeY; y++){
				double dX = (centerX - x)/rad;
				double dY = (centerY - y)/rad;
				double dist = hypot(dX, dY);
				double scale = exp(-dist*dist);
				if (xComp){
					scale *= dX/dist;
				}else if (yComp){
					scale *= dY/dist;
				}
				
				int index = dim.sizeX*y + x;
				data[index] += (float)(mag*scale);
			}
		}
	}
}
