#include "Calculator.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

CCalculator::CCalculator(int width, int height, float FOV)
	: m_width(width), m_height(height), m_FOV(FOV)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMallocPitch((void**)&m_d_buffer, &m_pitch, width * sizeof(float), height);

	if (cudaStatus != cudaSuccess) {
		ReactToCudaError(cudaStatus);
	}

}

CCalculator::~CCalculator()
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
