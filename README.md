# GPUFluid
GPU-based 2D incompressible fluid sim using OpenCL with JOCL Java wrapper

Fluid sim is 2D incompressible Navier-Stokes solver based on "Stable Fluids" 1999 paper by Jos Stam

The program is set up to generate graphics that look like space nebuas.

The Java code contains a high-level wrapper for OpenCL, and the implementation of the fluid sim using the wrapper.
Some image post-processing is done on the CPU. There is also a simple GUI to control rendering options.

The OpenCL kernels are in the "kernels" directory.

![](/nebula2.png?raw=true)
![](/nebula1.png?raw=true)
