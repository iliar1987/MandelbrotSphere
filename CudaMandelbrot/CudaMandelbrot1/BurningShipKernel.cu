#include "BurningShipKernel.h"


#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

#include "utilities.cuh"

#include "ComplexIterationKernel.cuh"

__device__ bool IterateBurningShip(IN OUT CComplexFP128 &z, IN const CComplexFP128 &c)
{
	if (((z.x.hihi & 0x80000000) >> 1) != (z.x.hihi & 0x40000000)
		|| ((z.y.hihi & 0x80000000) >> 1) != (z.y.hihi & 0x40000000))
		return true;

	if (z.x.IsNeg())
		z.x.Negate();
	if (z.y.IsNeg())
		z.y.Negate();

	CFixedPoint128 z_x_sqr = z.x.Sqr();
	CFixedPoint128 z_y_sqr = z.y.Sqr();

	CFixedPoint128 sumOfSquares(z_x_sqr);
	sumOfSquares += z_y_sqr;
	if (sumOfSquares.IsNeg())
		return true;

	z.y = z.x * z.y;
	z.y <<= 1;

	z.x = z_x_sqr;
	z.x -= z_y_sqr;

	z += c;

	return false;
}

typedef CComplexIterationTextureFiller<IterateBurningShip> CBurningShipTextureFiller;

CTextureFiller * CreateBurningShipTextureFiller(int width, int height, float FOV)
{
	return new CBurningShipTextureFiller(width, height, FOV);
}
