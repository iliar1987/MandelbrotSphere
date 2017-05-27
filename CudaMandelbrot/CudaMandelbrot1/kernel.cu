#include "kernel.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

__global__ void kern1(float4* buffer, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float4 *pixel;
	if (x >= width || y >= height) return;

	//pixel = (float4 *)(buffer + y*pitch) + 4 * x;
	pixel = buffer + (y * width + x);

	pixel->x = (float)x / width;
	pixel->y = (float)y / height;
	pixel->z = t;
	pixel->w = 1.0f;
}

SimpleFillTexture::SimpleFillTexture(int width, int height)
	: m_width(width), m_height(height)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMallocPitch((void**)&m_d_buffer, &m_pitch, width * sizeof(float4), height);

	if (cudaStatus != cudaSuccess) {
		ReactToCudaError(cudaStatus);
	}

}

SimpleFillTexture::~SimpleFillTexture()
{
	if (m_d_buffer)
	{
		cudaError_t status = cudaFree(m_d_buffer);
		if (status != cudaSuccess)
		{
			ReactToCudaError(status);
		}
	}
}

void SimpleFillTexture::UpdateBuffer()
{
	static float t = 0;
	t += 0.02;
	if (t >= 1)
		t = 0;
	dim3 Db = dim3(8, 8);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((GetWidth() + Db.x - 1) / Db.x, (GetHeight() + Db.y - 1) / Db.y);
	kern1 << <Dg, Db >> > (m_d_buffer, GetWidth(), GetHeight(), GetPitch(), t);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		ReactToCudaError(err);
	}
}