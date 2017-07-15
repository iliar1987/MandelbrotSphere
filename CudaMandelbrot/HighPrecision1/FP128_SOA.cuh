#pragma once

#include "FP128.cuh"
#include <cuda_runtime.h>

class CFP128_SOA
{
public:
	typedef CFixedPoint128 VALUE_TYPE;
	uint32_t *lolo=nullptr, *lohi= nullptr, *hilo= nullptr, *hihi= nullptr;
	__host__ cudaError_t Alloc(size_t size)
	{
		cudaError_t status;
		status = cudaMalloc<uint32_t>(&lolo,size);
		if (status != cudaSuccess)
		{
			return status;
		}
		status = cudaMalloc<uint32_t>(&lohi, size);
		if (status != cudaSuccess)
		{
			cudaFree(lolo);
			return status;
		}
		status = cudaMalloc<uint32_t>(&hilo, size);
		if (status != cudaSuccess)
		{
			cudaFree(lolo);
			cudaFree(lohi);
			return status;
		}
		status = cudaMalloc<uint32_t>(&hihi, size);
		if (status != cudaSuccess)
		{
			cudaFree(lolo);
			cudaFree(lohi);
			cudaFree(hilo);
			return status;
		}
		return cudaSuccess;
	}
	__host__ cudaError_t Free()
	{
		cudaError_t status;
		if (lolo)
		{
			status = cudaFree(lolo);
			if (status != cudaSuccess)
			{
				return status;
			}
		}
		if (lohi)
		{
			status = cudaFree(lohi);
			if (status != cudaSuccess)
			{
				return status;
			}
		}
		if (hilo)
		{
			status = cudaFree(hilo);
			if (status != cudaSuccess)
			{
				return status;
			}
		}
		if (hihi)
		{
			status = cudaFree(hihi);
			if (status != cudaSuccess)
			{
				return status;
			}
		}
		return cudaSuccess;
	}

	__host__ __device__ void GetValue(uint32_t ind, CFixedPoint128& val) const
	{
		val.lolo = lolo[ind];
		val.lohi = lohi[ind];
		val.hilo = hilo[ind];
		val.hihi = hihi[ind];
	}
	__host__ __device__ void SetValue(uint32_t ind, const CFixedPoint128& val) const
	{
		lolo[ind] = val.lolo;
		lohi[ind] = val.lohi;
		hilo[ind] = val.hilo;
		hihi[ind] = val.hihi;
	}
};

class CComplex128_SOA
{
public:
	typedef CComplexFP128 VALUE_TYPE;
	CFP128_SOA soa_x, soa_y;

	__host__ cudaError_t Alloc(size_t size)
	{
		cudaError_t status;
		status = soa_x.Alloc(size);
		if (status != cudaSuccess)
		{
			return status;
		}
		status = soa_y.Alloc(size);
		if (status != cudaSuccess)
		{
			soa_x.Free();
			return status;
		}
		return cudaSuccess;
	}
	__host__ cudaError_t Free()
	{
		cudaError_t status;
		status = soa_x.Free();
		if (status != cudaSuccess)
		{
			return status;
		}
		status = soa_y.Free();
		if (status != cudaSuccess)
		{
			return status;
		}
		return cudaSuccess;
	}

	__host__ __device__ void GetValue(uint32_t ind, CComplexFP128& val) const
	{
		soa_x.GetValue(ind, val.x);
		soa_y.GetValue(ind, val.y);
	}

	__host__ __device__ void SetValue(uint32_t ind, const CComplexFP128& val) const
	{
		soa_x.SetValue(ind, val.x);
		soa_y.SetValue(ind, val.y);
	}

};

typedef SOA2D<CComplex128_SOA> CComplex128_SOA2d;