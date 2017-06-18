#pragma once

#include "TextureFiller.h"


union Quaternion
{
	struct
	{
		float i, j, k, r;
	};
	float arr[4];
	__host__ __device__ Quaternion(float4 vec)
		: r(vec.w), i(vec.x), j(vec.y), k(vec.z)
	{
	}
	__host__ __device__ Quaternion(float i, float j, float k, float r)
		: i(i), j(j), k(k), r(r)
	{

	}

	__host__ __device__ Quaternion operator * (const Quaternion &q) const
	{
		return Quaternion(r * q.r - i * q.i - j * q.j - k * q.k,
			r * q.i + i * q.r + j * q.k - k * q.j,
			r * q.j + j * q.r + k * q.i - i * q.k,
			r * q.k + k * q.r + i * q.j - j * q.i);
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
		Quaternion q(pos.x, pos.y, pos.z, 0);
		q = operator*(q * Conj());
		return float3{ q.i, q.j, q.k };
	}
};


__host__ __device__ inline float3 operator * (float x, float3 v)
{
	return{ x*v.x, x*v.y, x*v.z };
}

__host__ __device__ inline float3 operator * (float3 v, float x)
{
	return{ x*v.x, x*v.y, x*v.z };
}

__host__ __device__ inline float3 operator + (float3 u, float3 v)
{
	return{ u.x + v.x,u.y + v.y,u.z + v.z };
}


__device__ inline void GetThetaPhi(float& theta, float &phi, const int x, const int y, const CTextureFiller::KernelParameters &params)
{
	float3 posCamera = (float)(x - params.width / 2) * params.tFrameParams.vCamRight + (float)(y - params.height / 2) * params.tFrameParams.vCamUp + params.tFrameParams.vCamForward * (float)params.L;

	const float r = sqrtf(posCamera.x * posCamera.x + posCamera.y * posCamera.y);

	theta = atan2f(r, posCamera.z);
	phi = atan2f(posCamera.y, posCamera.x);
}