#include "kernel.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

#define ARR_AS_FLOAT3(arr) (*((float3*) (arr) ))

union Quaternion
{
	struct
	{
		float i,j,k,r;
	};
	float arr[4];
	__host__ __device__ Quaternion(float4 vec)
		: r(vec.w), i(vec.x),j(vec.y),k(vec.z)
	{
	}
	__host__ __device__ Quaternion(float i, float j, float k, float r)
		: i(i),j(j),k(k),r(r)
	{

	}

	__host__ __device__ Quaternion operator * (const Quaternion &q) const
	{
		return Quaternion(r * q.r		- i * q.i - j * q.j - k * q.k,
			r * q.i + i * q.r			+ j * q.k - k * q.j,
			r * q.j + j * q.r			+ k * q.i - i * q.k,
			r * q.k + k * q.r			+ i * q.j - j * q.i);
	}

	__host__ __device__ Quaternion operator + (const Quaternion &other) const
	{
		return Quaternion(r + other.r,
			i + other.i,
			j + other.j,
			k + other.k);
	}

	__host__ __device__ Quaternion Conj() const
	{
		return Quaternion(r, -i, -j, -k);
	}

	__host__ __device__ float3 RotateVector(float3 pos) const
	{
		Quaternion q(pos.x, pos.y, pos.z,0);
		q = operator*(q * Conj());
		return float3 { q.i, q.j, q.k };
	}
};


__host__ __device__ float3 operator * (float x, float3 v)
{
	return{ x*v.x, x*v.y, x*v.z };
}

__host__ __device__ float3 operator * (float3 v, float x)
{
	return{ x*v.x, x*v.y, x*v.z };
}

__host__ __device__ float3 operator + (float3 u, float3 v)
{
	return{ u.x + v.x,u.y + v.y,u.z + v.z };
}
	

__global__ void kernSpherical(float4* buffer, const int width, const int height, const size_t pitch, const float t, const float L, const float3 vCameraRight, const float3 vCameraUp, const float3 vCameraForward)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	float4 *pixel = buffer + (y * width + x);
	
	float3 posCamera = (float)(x - width / 2) * vCameraRight + (float)(y - height / 2) * vCameraUp + vCameraForward * (float)L;

	const float r = sqrtf(posCamera.x * posCamera.x + posCamera.y * posCamera.y);

	const float theta = atan2f(posCamera.z,r);
	const float phi = atan2f(posCamera.y, posCamera.x);
	
	pixel->x = (1.0f + cosf(phi))*0.5f;
	pixel->y = theta / PIf + 0.5f;
	pixel->z = (1.0f + sinf(phi))*0.5f;;
	pixel->w = 1.0f;
}



SimpleFillTexture::SimpleFillTexture(int width, int height, float FOV)
	: m_width(width), m_height(height), m_FOV(FOV)
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

	float L = (float)m_height / 2.0f / tanf(m_FOV / 2.0f);
	dim3 Db = dim3(8, 8);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((GetWidth() + Db.x - 1) / Db.x, (GetHeight() + Db.y - 1) / Db.y);

	kernSpherical << <Dg, Db >> > (m_d_buffer, GetWidth(), GetHeight(), GetPitch(), t,L, ARR_AS_FLOAT3( vCamRight), ARR_AS_FLOAT3( vCamUp), ARR_AS_FLOAT3( vCamForward));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		ReactToCudaError(err);
	}
}