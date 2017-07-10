#include "TextureFiller.h"

#include "TextureInfo.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

#include "assert.h"

#include "..\HighPrecision1\FP128.cuh"

CTextureFiller::CTextureFiller(int width, int height, float FOV)
	: m_width(width), m_height(height),
	m_fL(height / 2.0f / tanf(FOV / 2.0f))
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMallocPitch((void**)&m_d_buffer, &m_pitch, width * sizeof(float), height);

	if (cudaStatus != cudaSuccess) {
		ReactToCudaError(cudaStatus);
	}

	m_poleCoords.x = new CFixedPoint128{ 0,0 };
	m_poleCoords.y = new CFixedPoint128{ 0,0 };
}

CTextureFiller::~CTextureFiller()
{
	delete m_poleCoords.x;
	delete m_poleCoords.y;
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

void CTextureFiller::UpdateBuffer(const FrameParameters &params)
{
	KernelParameters kParams;
	kParams.tFrameParams = params;
	kParams.width = GetWidth();
	kParams.height = GetHeight();
	kParams.L = GetL();
	kParams.pitch = GetPitch();

	LaunchKernel(kParams);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		ReactToCudaError(err);
	}
}

void CTextureFiller::PoleCoordsGet(float& x, float &y)
{
	x = static_cast<float>(*m_poleCoords.x);
	y = static_cast<float>(*m_poleCoords.y);
}

void CTextureFiller::PoleCoordsAdd(float dx, float dy)
{
	*m_poleCoords.x += CFixedPoint128(dx);
	*m_poleCoords.y += CFixedPoint128(dy);
}

void CTextureFiller::PoleCoordsSet(float x, float y)
{
	*m_poleCoords.x = CFixedPoint128(x);
	*m_poleCoords.y = CFixedPoint128(y);
}

void CTextureFiller::PoleCoordsZoom(float3 vForward, float rho, float rho_new)
{
	float temp = sqrtf(vForward.x * vForward.x + vForward.y * vForward.y);
	float theta = atan2f(temp, vForward.z);
	float phi = atan2f(vForward.y, vForward.x);

	float rho_delta = rho_new - rho;

	float dr = 2 * rho_delta * tanf(theta / 2);
	float dx = dr * cosf(phi);
	float dy = dr * sinf(phi);

	*m_poleCoords.x += CFixedPoint128(-dx);
	*m_poleCoords.y += CFixedPoint128(-dy);
}
