#pragma once

#include "CComplex.cuh"

template<typename Real,typename Real_SOA>
class CComplex_SOA
{
private:
	union UArrays
	{
		struct
		{
			Real_SOA soa_x, soa_y;
		};
		Real_SOA arr[2];
		UArrays()
		{
			memset(arr, 0, sizeof(arr));
		}
	} m_arrays;

public:
	__host__ cudaError_t Alloc(int length)
	{
		cudaError_t status;
		status = m_arrays.soa_x.Alloc(length);
		if (status != cudaSuccess)
		{
			return status;
		}
		status = m_arrays.soa_y.Alloc(length);
		if (status != cudaSuccess)
		{
			return status;
		}
		return cudaSuccess;
	}
	__host__ cudaError_t Free()
	{
		cudaError_t status;
		status = m_arrays.soa_x.Free();
		if (status != cudaSuccess)
		{
			return status;
		}
		status = m_arrays.soa_y.Free();
		if (status != cudaSuccess)
		{
			return status;
		}
		return cudaSuccess;
	}

	template<typename Complex>
	__host__ __device__ void GetValue(uint32_t ind, Complex& val) const
	{
		m_arrays.soa_x.GetValue(ind, val.x);
		m_arrays.soa_y.GetValue(ind, val.y);
	}

	template<typename Complex>
	__host__ __device__ void SetValue(uint32_t ind, const Complex& val) const
	{
		m_arrays.soa_x.SetValue(ind, val.x);
		m_arrays.soa_y.SetValue(ind, val.y);
	}

	__host__ cudaError_t CopyFromAsync(const CComplex_SOA<Real,Real_SOA> &other, int length, cudaStream_t stream)
	{
		cudaError_t status;
		for (int i = 0;i < 2;++i)
		{
			status = m_arrays.arr[i].CopyFromAsync(other.m_arrays.arr[i], length, stream);
			if (status != cudaSuccess)
			{
				return status;
			}
		}
		return cudaSuccess;
	}
};
