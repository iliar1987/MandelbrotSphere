#pragma once

#include "FP128.cuh"
#include "FP64.cuh"
#include "FP32.cuh"
#include <cuda_runtime.h>
#include "CComplex_SOA.cuh"

class CFP128_SOA
{
private:
	typedef CFixedPoint128 VALUE_TYPE;
	union UArrays
	{
		struct
		{
			uint32_t *lolo, *lohi, *hilo, *hihi;
		};
		uint32_t* arr[4];
		UArrays()
		{
			memset(arr, 0, sizeof(*this));
		}
	} m_arrays;

public:
	__host__ cudaError_t Alloc(int length)
	{
		cudaError_t status;
		for (int i = 0;i < 4;++i)
		{
			status = cudaMalloc(&m_arrays.arr[i], length * sizeof(uint32_t));
			if (status != cudaSuccess)
			{
				return status;
			}
		}
		return cudaSuccess;
	}
	__host__ cudaError_t Free()
	{
		cudaError_t status;
		for (int i = 0;i < 4;++i)
		{
			if (m_arrays.arr[i])
			{
				status = cudaFree(m_arrays.arr[i]);
				if (status != cudaSuccess)
				{
					return status;
				}
			}
		}
		return cudaSuccess;
	}

	__device__ void GetValue(uint32_t ind, CFixedPoint128& val) const
	{
		val.lolo = m_arrays.lolo[ind];
		val.lohi = m_arrays.lohi[ind];
		val.hilo = m_arrays.hilo[ind];
		val.hihi = m_arrays.hihi[ind];
	}
	__device__ void SetValue(uint32_t ind, const CFixedPoint128& val) const
	{
		m_arrays.lolo[ind] = val.lolo;
		m_arrays.lohi[ind] = val.lohi;
		m_arrays.hilo[ind] = val.hilo;
		m_arrays.hihi[ind] = val.hihi;
	}

	__device__ void GetValue(uint32_t ind, CFixedPoint64& val) const
	{
		val.m_lo = m_arrays.hilo[ind];
		val.m_hi = m_arrays.hihi[ind];
	}

	__device__ void SetValue(uint32_t ind, const CFixedPoint64& val) const
	{
		m_arrays.hilo[ind] = val.m_lo;
		m_arrays.hihi[ind] = val.m_hi;
	}

	__device__ void GetValue(uint32_t ind, CFixedPoint32& val) const
	{
		val.m_val32 = m_arrays.hihi[ind];
	}

	__device__ void SetValue(uint32_t ind, const CFixedPoint32& val) const
	{
		m_arrays.hihi[ind] = val.m_val32;
	}

	__host__ cudaError_t CopyFromAsync(const CFP128_SOA &other,int length, cudaStream_t stream)
	{
		cudaError_t status;
		for (int i = 0;i < 4;++i)
		{
			status = cudaMemcpyAsync(m_arrays.arr[i], other.m_arrays.arr[i], length * sizeof(uint32_t), cudaMemcpyDeviceToDevice,stream);
			if (status != cudaSuccess)
			{
				return status;
			}
		}
		return cudaSuccess;
	}

	//__host__ cudaError_t SetZero(int size)
	//{
	//	cudaError_t result;
	//	cudaMemset(lolo
	//	return result;
	//}
};

typedef CComplex_SOA<CFixedPoint128, CFP128_SOA> CComplex128_SOA;
