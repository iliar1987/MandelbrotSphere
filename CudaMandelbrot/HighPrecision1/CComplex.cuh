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

	__device__ __host__ CComplex(Real x, Real y) : x(x), y(y)
	{}

	__device__ __host__ CComplex(std::pair<Real, Real> p)
		: CComplex(p.first,p.second)
	{}

	__device__ __host__ CComplex(float x,float y) : x(x),y(y)
	{}

	__device__ __host__ CComplex(std::pair<float,float> p)
		: CComplex(p.first,p.second)
	{}

	__device__ __host__ CComplex& operator += (const CComplex& other)
	{
		x += other.x;
		y += other.y;
		return *this;
	}
	__device__ __host__ CComplex& operator <<= (const unsigned int shft)
	{
		x <<= shft;
		y <<= shft;
		return *this;
	}
	__device__ __host__ CComplex& operator >>= (const unsigned int shft)
	{
		x >>= shft;
		y >>= shft;
		return *this;
	}

	//__device__ __host__ CComplex operator * (const CComplex& b) const
	//{
	//	CComplex result;

	//	result.x = x * b.x;
	//	result.x -= y * b.y;

	//	result.y = x*b.y;
	//	result.y += y*b.x;

	//	return result;
	//}

	__device__  CComplex& operator *= (const Real& r)
	{
		x *= r;
		y *= r;
		return *this;
	}

	__device__  CComplex operator * (const Real& r) const
	{
		CComplex result(*this);
		result *= r;
		return result;
	}

	__device__  void Sqr()
	{
		Real oldX(x);
		Real oldY(y);

		x.Sqr();
		y.Sqr();
		x -= y;
		

		y = oldX;
		y *= oldY;
		y <<= 1;
	}
};

template<typename Real>
std::ostream& operator << (std::ostream &o, const CComplex<Real>& C)
{
	o << "{ " << C.x << ", " << C.y << " }";
	return o;
}
