import static java.lang.Math.*;
import static Vector.Vector.*;
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
import java.util.concurrent.*;
import java.awt.image.*;
import javax.imageio.*;
import java.io.*;

public class FluidGUI{
	
	double[] LIGHT_SOURCE = normalized(new double[]{-1.2, -1, 0.3});
	
	enum DisplayMode{
		VEL,
		DYE,
		PRESSURE
	}
	
	JPanel panel;
	DisplayPanel display;
	JLabel iterLabel, timingLabel;
	JComboBox<DisplayMode> displayBox;
	JTextField scaleField, framerateField;
	JCheckBox gradBox;
	
	JSlider l1max, l2max, l1val, l2val, l3max, l3val;
	JSlider gradMag, gradDir;
	JTextField gradMult, dirPow;
	
	Fluid2D fluid;
	int iter, dataIter;
	boolean running, rendering;
	CountDownLatch pauseLatch;
	long lastFrameTime;
	int lastFrameIter;
	int iterPerFrame;
	int pauseIter;
	float[] scaledVal;
	BufferedImage background;
	BufferedImage fluidImage;
	Color randColor;
	boolean separateOptions;
	int pixelSize;
	
	public FluidGUI(Fluid2D fluid){
		
		//System.setProperty("sun.java2d.opengl","True");
		
		this.fluid = fluid;
		scaledVal = new float[fluid.data.length];
		int sizeX = fluid.dim.sizeX;
		separateOptions = true;
		pixelSize = 1;
		
		pauseIter = 4000 + (int)(random()*3000);
		randomizeColor();
		
		try{
			background = ImageIO.read(new File("background.png"));
		}catch(IOException e){
			background = new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB);
		}
		
		fluidImage = new BufferedImage(fluid.dim.sizeX, fluid.dim.sizeY, BufferedImage.TYPE_INT_ARGB);
		
		WindowListener exitListener = new WindowListener(){
			public void windowClosing(WindowEvent arg0){
				running = false;
				if (pauseLatch != null)
					pauseLatch.countDown();
			}
			public void windowActivated(WindowEvent arg0){}
			public void windowClosed(WindowEvent arg0){}
			public void windowDeactivated(WindowEvent arg0){}
			public void windowDeiconified(WindowEvent arg0){}
			public void windowIconified(WindowEvent arg0){}
			public void windowOpened(WindowEvent arg0){}
		};
		
		display = new DisplayPanel();
		
		iterLabel = new JLabel();
		iterLabel.setPreferredSize(new Dimension(100, 15));
		timingLabel = new JLabel();
		timingLabel.setPreferredSize(new Dimension(115, 15));
		
		displayBox = new JComboBox<DisplayMode>();
		displayBox.setPreferredSize(new Dimension(100, 20));
		for (DisplayMode mode : DisplayMode.values())
			displayBox.addItem(mode);
		displayBox.setSelectedItem(DisplayMode.DYE);
		displayBox.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				if (pauseLatch != null){
					copyDisplayData();
					refresh();
				}
			}
		});
		
		scaleField = new JTextField("1.0");
		scaleField.setPreferredSize(new Dimension(50, 20));
		scaleField.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				if (pauseLatch != null)
					refresh();
			}
		});
		
		JButton pauseButton = new JButton("Pause");
		pauseButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				pause();
				refresh();
			}
		});
		JButton unpauseButton = new JButton("Unpause");
		unpauseButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				unpause();
			}
		});
		
		iterPerFrame = 2;
		framerateField = new JTextField(String.valueOf(iterPerFrame));
		framerateField.setPreferredSize(new Dimension(50, 20));
		framerateField.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				try{
					iterPerFrame = Integer.parseInt(framerateField.getText());
				}catch (NumberFormatException ex){
					iterPerFrame = 2;
				}
			}
		});
		
		gradBox = new JCheckBox("grad");
		
		JButton colorButton = new JButton("Color");
		colorButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				randomizeColor();
				if (pauseLatch != null)
					refresh();
			}
		});
		
		JButton saveButton = new JButton("Save");
		saveButton.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e){
				save();
			}
		});
		
		JPanel buttonPanel = new JPanel();
		buttonPanel.setPreferredSize(new Dimension(sizeX, 40));
		buttonPanel.add(iterLabel);
		buttonPanel.add(timingLabel);
		buttonPanel.add(displayBox);
		buttonPanel.add(new JLabel("  scale:"));
		buttonPanel.add(scaleField);
		buttonPanel.add(pauseButton);
		buttonPanel.add(unpauseButton);
		buttonPanel.add(new JLabel("  IPF:"));
		buttonPanel.add(framerateField);
		buttonPanel.add(gradBox);
		buttonPanel.add(colorButton);
		buttonPanel.add(saveButton);
		
		l1max = new JSlider(0, 100, 42 + (int)(3*random())-2);
		l2max = new JSlider(0, 100, 53 + (int)(5*random())-2);
		l3max = new JSlider(0, 100, 98);
		l1val = new JSlider(0, 100, 0);
		l2val = new JSlider(0, 100, 45 + (int)(8*random()));
		l3val = new JSlider(0, 100, 100);
		
		gradMag = new JSlider(0, 100, 0);
		gradDir = new JSlider(0, 100, 0);
		gradMult = new JTextField("0.1");
		gradMult.setPreferredSize(new Dimension(50, 20));
		dirPow = new JTextField("0.25");
		dirPow.setPreferredSize(new Dimension(50, 20));
		
		JPanel renderPanel = new JPanel();
		renderPanel.setPreferredSize(new Dimension(780, 120));
		renderPanel.add(new JLabel("Opacity ranges:"));
		renderPanel.add(l1max);
		renderPanel.add(l2max);
		renderPanel.add(l3max);
		renderPanel.add(new JLabel("Opacity levels: "));
		renderPanel.add(l1val);
		renderPanel.add(l2val);
		renderPanel.add(l3val);
		renderPanel.add(new JLabel("Gradient dependence:"));
		renderPanel.add(gradMag);
		renderPanel.add(gradDir);
		renderPanel.add(gradMult);
		renderPanel.add(dirPow);
		
		panel = new JPanel();
		panel.setPreferredSize(new Dimension(sizeX + 10, fluid.dim.sizeY + 140));
		panel.add(display);
		
		if (separateOptions){
			JPanel separatePanel = new JPanel();
			separatePanel.setPreferredSize(new Dimension(1000, 140));
			separatePanel.add(buttonPanel);
			separatePanel.add(renderPanel);
			JFrame separateFrame = new JFrame("Controls");
			separateFrame.add(separatePanel);
			separateFrame.pack();
			separateFrame.setVisible(true);
		}else{
			panel.add(buttonPanel);
			panel.add(renderPanel);
		}
		
		JFrame frame = new JFrame("Fluid");
		frame.addWindowListener(exitListener);
		frame.add(panel);
		frame.pack();
		frame.setVisible(true);
		
		running = true;
		iter = 0;
		new Thread("Run thread"){
			public void run(){
				try{
					Thread.sleep(500);
				}catch (Exception ex){}
				FluidGUI.this.run();
			}
		}.start();
	}
	
	private void run(){
		while (running){
			CountDownLatch latch = pauseLatch;
			if (latch != null){
				try{
					pauseLatch.await();
				}catch (InterruptedException ex){}
			}
			
			iter++;
			fluid.iterate();
			
			if (iter%iterPerFrame == 0)
				refresh();
			
			if (iter == pauseIter){
				gradDir.setValue(25 + (int)(random()*30));
				pause();
			}
		}
		
		try{
			Thread.sleep(100);
		}catch (Exception ex){}
		fluid.dispose();
		System.exit(0);
	}
	
	private void copyDisplayData(){
		dataIter = iter;
		DisplayMode mode = (DisplayMode)displayBox.getSelectedItem();
		if (mode == DisplayMode.VEL){
			fluid.copyBack(Fluid2D.VX_INDEX);
		}else if (mode == DisplayMode.DYE){
			fluid.copyBack(Fluid2D.DYE_INDEX);
		}else if (mode == DisplayMode.PRESSURE){
			fluid.copyBack(Fluid2D.P_INDEX);
		}
	}
	
	private void pause(){
		if (pauseLatch == null)
			pauseLatch = new CountDownLatch(1);
	}
	
	private void unpause(){
		if (pauseLatch != null){
			pauseLatch.countDown();
			pauseLatch = null;
		}
	}
	
	public void refresh(){
		if (!rendering){
			rendering = true;
			
			long frameTime = System.currentTimeMillis();
			int diffTime = (int)(frameTime - lastFrameTime);
			int diffIter = (int)(iter - lastFrameIter);
			double timePerFrame = (int)(diffTime*10.0/diffIter)/10.0;
			
			lastFrameTime = frameTime;
			lastFrameIter = iter;
			
			copyDisplayData();
			iterLabel.setText("Iteration: " + iter);
			
			if (timePerFrame > 0 && timePerFrame < 1e6)
				timingLabel.setText("Timing: " + timePerFrame + " ms");
			
			new Thread("FluidImageThread"){
				public void run(){
					try{
						generateFluidImage();
					}catch (Exception ex){
						ex.printStackTrace();
					}
					panel.repaint();
					rendering = false;
				}
			}.start();
		}
	}
	
	private void generateFluidImage(){
		double scale = 1.0;
		try{
			scale = Double.parseDouble(scaleField.getText());
		}catch (NumberFormatException ex){}
		
		double gradScale = 1.0;
		try{
			gradScale = Double.parseDouble(gradMult.getText());
		}catch (NumberFormatException ex){}
		
		double dirPow = 1.0;
		try{
			dirPow = Double.parseDouble(FluidGUI.this.dirPow.getText());
		}catch (NumberFormatException ex){}
		
		for (int x = 0; x < fluid.dim.sizeX; x++){
			for (int y = 0; y < fluid.dim.sizeY; y++){
				int index = y*fluid.dim.sizeX + x;
				
				float val;
				if (gradBox.isSelected()){
					val = (float)gradMag(fluid.data, x, y);
				}else
					val = fluid.data[index];
				scaledVal[index] = (float)getGUIAlpha(scale*val);
			}
		}
		
		Graphics2D g2d = fluidImage.createGraphics();
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR));
		g2d.fillRect(0, 0, fluidImage.getWidth(), fluidImage.getHeight());
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER));
		
		for (int x = 0; x < fluid.dim.sizeX; x++){
			for (int y = 0; y < fluid.dim.sizeY; y++){
				int index = y*fluid.dim.sizeX + x;
				
				int posX = x*pixelSize;
				int posY = (fluid.dim.sizeY - y - 1)*pixelSize;
				
				Color color = new Color(
						randColor.getRed(), randColor.getGreen(), randColor.getBlue(),
						(int)(scaledVal[index]*255));
				g2d.setColor(color);
				g2d.fillRect(posX, posY, pixelSize, pixelSize);
				
				double mag = gradMag(scaledVal, x, y)*gradMag.getValue()/100.0;
				if (mag > 0){
					double magMult = gradScale*sqrt(mag);
					Color boundary = new Color(
							randColor.getRed(), randColor.getGreen(), randColor.getBlue(),
							(int)(magMult*255));
					g2d.setColor(boundary);
					g2d.fillRect(posX, posY, pixelSize, pixelSize);
				}
				
				double dir = max(0.0, gradDir(scaledVal, x, y)*gradDir.getValue()/100.0);
				if (dir > 0){
					double dirMult = max(0, min(1, gradScale*pow(dir, dirPow)));
					Color light = new Color(200, 200, 250, (int)(dirMult*255));
					g2d.setColor(light);
					g2d.fillRect(posX, posY, pixelSize, pixelSize);
				}
				
				
				//boolean signed = displayBox.getSelectedItem() != DisplayMode.DYE;
				//g2d.setColor(getColor(val, signed, scale));
				//g2d.fillRect(posX, posY, pixelSize, pixelSize);
			}
		}
		
		g2d.dispose();
	}
	
	public class DisplayPanel extends JComponent{
		
		public DisplayPanel(){
			this.setPreferredSize(new Dimension(fluid.dim.sizeX, fluid.dim.sizeY));
			
		}
		
		public void paint(Graphics graphics){
			
			graphics.setColor(Color.BLACK);
			graphics.fillRect(0, 0, fluid.dim.sizeX*pixelSize, fluid.dim.sizeY*pixelSize);
			graphics.drawImage(background, 0, 0, null);
			graphics.drawImage(fluidImage, 0, 0, null);
		}
	}
	
	private double getGUIAlpha(double val){
		double l1max = this.l1max.getValue()/100.0;
		double l2max = this.l2max.getValue()/100.0;
		double l3max = this.l3max.getValue()/100.0;
		double l1val = this.l1val.getValue()/100.0;
		double l2val = this.l2val.getValue()/100.0;
		double l3val = this.l3val.getValue()/100.0;
		
		double alpha;
		if (val < l1max){
			alpha = l1val*(val/l1max);
		}else if (val < l2max){
			double ratio = pow((val - l1max)/(l2max - l1max), 2);
			alpha = l1val*(1.0 - ratio) + l2val*ratio;
		}else if (val < l3max){
			double ratio = (val - l2max)/(l3max - l1max);
			alpha = l2val*(1.0 - ratio) + l3val*ratio;
		}else{
			double ratio = (val - l3max)/(1.0 - l3max);
			alpha = l3val*(1.0 - ratio) + 1.0*ratio;
		}
		
		return alpha;
	}
	
	static final int BORDER_SIZE = 10;
	private void save(){
		BufferedImage borderless = fluidImage.getSubimage(BORDER_SIZE, BORDER_SIZE,
				fluidImage.getWidth()-2*BORDER_SIZE, fluidImage.getHeight()-2*BORDER_SIZE);
		try{
			ImageIO.write(borderless, "png", new File("saved/" + (int)(100000 + random()*900000) + ".png"));
		}catch (IOException e){
			e.printStackTrace();
		}
	}
	
	private void randomizeColor(){
		int lightness = (int)(pow(random(), 2.0)*50);
		randColor = new Color(
				lightness+0+(int)(pow(random(), 4.0)*30),
				lightness+0+(int)(pow(random(), 4.0)*20),
				lightness+0+(int)(pow(random(), 4.0)*20));
	}
	
	private double gradMag(float[] data, int x, int y){
		if (y == 0 || y == fluid.dim.sizeY-1 || x == 0 || x == fluid.dim.sizeX-1)
			return 0.0;
		
		int index = y*fluid.dim.sizeX + x;
		double dx = (data[index+1] - data[index-1])/2;
		double dy = (data[index+fluid.dim.sizeX] - data[index-fluid.dim.sizeX])/2;
		return sqrt(dx*dx + dy*dy);
	}
	
	private double gradDir(float[] data, int x, int y){
		if (y == 0 || y == fluid.dim.sizeY-1 || x == 0 || x == fluid.dim.sizeX-1)
			return 0.0;
		
		int index = y*fluid.dim.sizeX + x;
		double dx = (data[index+1] - data[index-1])/2;
		double dy = (data[index-fluid.dim.sizeX] - data[index+fluid.dim.sizeX])/2;
		return dx*LIGHT_SOURCE[0] + dy*LIGHT_SOURCE[1];
	}
}
