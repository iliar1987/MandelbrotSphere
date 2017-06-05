#include "SimpleFillTexture.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

#include "../HighPrecision1/FP128.cuh"

#include "utilities.cuh"


__global__ void kernSpherical(float* buffer, const int width, const int height, const size_t pitch, const float t, const float L, const float3 vCameraRight, const float3 vCameraUp, const float3 vCameraForward)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	float *pixel = buffer + (y * width + x);
	
	float3 posCamera = (float)(x - width / 2) * vCameraRight + (float)(y - height / 2) * vCameraUp + vCameraForward * (float)L;

	const float r = sqrtf(posCamera.x * posCamera.x + posCamera.y * posCamera.y);

	const float theta = atan2f(posCamera.z,r);
	const float phi = atan2f(posCamera.y, posCamera.x);
	
	*pixel = pow(sin(theta), 3) * (cosf(3.0f*(phi + t))) * 0.5f + 0.5f;
}

void SimpleFillTexture::UpdateBuffer(float vCamRight[3], float vCamUp[3], float vCamForward[3])
{
	static int dir = 1;
	static float t = 0;
	t += 0.02f * dir;
	if (t > 1)
	{
		t = 1;
		dir = -1;
	}
	else if (t < 0)
	{
		dir = 1;
		t = 0;
	}

	float L = (float)GetHeight() / 2.0f / tanf(GetFov() / 2.0f);
	dim3 Db = dim3(8, 8);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((GetWidth() + Db.x - 1) / Db.x, (GetHeight() + Db.y - 1) / Db.y);

	kernSpherical <<< Dg, Db >>> (GetBuffer(), GetWidth(), GetHeight(), GetPitch(), t,L, ARR_AS_FLOAT3( vCamRight), ARR_AS_FLOAT3( vCamUp), ARR_AS_FLOAT3( vCamForward));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		ReactToCudaError(err);
	}
}