#include "TextureFiller.h"

#include "TextureInfo.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

#include "assert.h"

CTextureFiller::CTextureFiller(int width, int height, float FOV)
	: m_width(width), m_height(height), m_FOV(FOV)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMallocPitch((void**)&m_d_buffer, &m_pitch, width * sizeof(float), height);

	if (cudaStatus != cudaSuccess) {
		ReactToCudaError(cudaStatus);
	}

}

CTextureFiller::~CTextureFiller()
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

void CTextureFiller::FillTexture(CTextureInfo& tex)
{
	assert(tex.GetElementSize() == sizeof(float));
	cudaError_t status;

	cudaArray *cuArray;

	cudaGraphicsResource* resources[] = { tex.GetCudaResource() };
	status = cudaGraphicsMapResources(1, resources);
	if (status != cudaSuccess)
	{
		ReactToCudaError(status);
	}

	status = cudaGraphicsSubResourceGetMappedArray(&cuArray, tex.GetCudaResource(), 0, 0);
	if (status != cudaSuccess)
	{
		ReactToCudaError(status);
	}

	status = cudaMemcpy2DToArray(cuArray, 0, 0, GetBuffer(), GetPitch(), GetWidth() * sizeof(float), GetHeight(), cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess)
	{
		ReactToCudaError(status);
	}

	status = cudaGraphicsUnmapResources(1, resources);
	if (status != cudaSuccess)
	{
		ReactToCudaError(status);
	}

}
