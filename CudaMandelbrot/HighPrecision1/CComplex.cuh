#pragma once

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <assert.h>
#include <intrin.h>

typedef unsigned long long int uint64_t;
typedef signed long long int int64_t;

template <typename Real>
class CComplex
{
public:
	Real x, y;
	__device__ __host__ CComplex() {}

	__device__ __host__ CComplex& operator = (const CComplex& other)
	{
		x = other.x;
		y = other.y;
		return *this;
	}

	__device__ __host__ CComplex(const Real &rx, const Real &ry) : x(rx), y(ry)
	{}

	__device__ __host__ CComplex(std::pair<const Real&, const Real&> p)
		: CComplex(p.first,p.second)
	{}

	__device__ __host__ CComplex(float fx,float fy) : x(fx),y(fy)
	{}

	__device__ __host__ CComplex(std::pair<float,float> p)
		: CComplex(p.first,p.second)
	{}

	__device__  CComplex& operator += (const CComplex& other)
	{
		x += other.x;
		y += other.y;
		return *this;
	}
	__device__ CComplex& operator <<= (const unsigned int shft)
	{
		x <<= shft;
		y <<= shft;
		return *this;
	}
	__device__ CComplex& operator >>= (const unsigned int shft)
	{
		x >>= shft;
		y >>= shft;
		return *this;
	}

	__device__ CComplex operator * (CComplex& b)
	{
		CComplex result;

		result.x = x * b.x;
		result.x -= y * b.y;

		result.y = x*b.y;
		result.y += y*b.x;

		return result;
	}

	/*__device__  CComplex operator * (const Real& r) const
	{
		return CComplex(x * r, y * r);
	}*/

	__device__  CComplex Sqr()
	{
		//return (*this) * (*this);
		CComplex result;
		result.x = x.Sqr();
		result.x -= y.Sqr();

		result.y = x*y;
		result.y <<= 1;
		
		return result;
	}
};

template<typename Real>
std::ostream& operator << (std::ostream &o, const CComplex<Real>& C)
{
	o << "{ " << C.x << ", " << C.y << " }";
	return o;
}
