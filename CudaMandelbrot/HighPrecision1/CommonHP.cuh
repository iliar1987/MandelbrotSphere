#pragma once


union UFUI32 {
	float f;
	uint32_t ui;
	//UFUI32(uint32_t x) : ui(x) {}
	//UFUI32(float x) : f(x) {}
};

__host__ __device__ inline uint32_t FloatToBin32(float f)
{
#ifndef __CUDA_ARCH__
	UFUI32 temp;
	temp.f = f;
	return temp.ui;
#else
	return __float_as_uint(f);
#endif
}


__host__ __device__ inline float Bin32ToFloat(uint32_t ui)
{
#ifndef __CUDA_ARCH__
	UFUI32 temp;
	temp.ui = ui;
	return temp.f;
#else
	return __uint_as_float(ui);
#endif
}

typedef unsigned long long int uint64_t;
typedef signed long long int int64_t;
