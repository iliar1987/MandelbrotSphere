# MandelbrotSphere
Rendering a mandelbrot fractal on a sphere projection.
The project works on unity + windows + directx11 + CUDA. (very limited, I know :)
At first I was using calculations inside a shader. But I hit the precision limitation of float32 very fast.
And so I made a DLL that uses CUDA to make 128 bits fixed point precision calculations.
Currently it is working (tag v2.1).

I am currently working on the following things:
1. Making a better frame rate for high number of iterations.
2. Making it work with VR.
