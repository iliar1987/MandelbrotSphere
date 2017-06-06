#include "SimpleFillTexture.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

#include "utilities.cuh"

__global__ void kernSpherical(float* buffer, CTextureFiller::KernelParameters params)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= params.width || y >= params.height) return;

	float *pixel = buffer + (y * params.width + x);

	float theta, phi;
	GetThetaPhi(theta, phi, x, y, params);

	*pixel = pow(sin(theta), 3) * (cosf(3.0f*(phi + params.tFrameParams.t))) * 0.5f + 0.5f;
}

void SimpleFillTexture::LaunchKernel(const KernelParameters& params)
{
	dim3 Db = dim3(8, 8);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((params.width + Db.x - 1) / Db.x, (params.height + Db.y - 1) / Db.y);

	kernSpherical <<< Dg, Db >>> (GetBuffer(),params);
}