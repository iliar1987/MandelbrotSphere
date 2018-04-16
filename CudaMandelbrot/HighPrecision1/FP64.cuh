#pragma once

#include <intrin.h>
#include <iostream>

#include "CommonHP.cuh"

class CFixedPoint64
{
public:
	union
	{
		int64_t m_val64;
		struct
		{
			uint32_t m_lo, m_hi;
		};
	};

	__device__ __host__ CFixedPoint64() {}
	__device__ __host__ explicit CFixedPoint64(int64_t v64)
		: m_val64(v64)
	{}

	__host__ __device__ inline explicit operator float() const;
	__host__ __device__ inline explicit CFixedPoint64(const float d);

	__host__ __device__ inline CFixedPoint64 operator * (const CFixedPoint64 &other);

	__device__ __host__ inline CFixedPoint64 & operator += (const CFixedPoint64 &other)
	{ 
		m_val64 += other.m_val64;
		return *this;
	}

	__device__ __host__ inline CFixedPoint64 & operator -= (const CFixedPoint64 &other)
	{
		m_val64 -= other.m_val64;
		return *this;
	};

	__device__ __host__ inline CFixedPoint64 & operator <<= (const unsigned char n) { m_val64 <<= n; return *this; }
	__device__ __host__ inline CFixedPoint64 & operator >>= (const unsigned char n) { m_val64 >>= n; return *this; }

	__device__ __host__ bool operator == (const CFixedPoint64& other) const
	{
		return m_val64 == other.m_val64;
	}

	__device__ __host__ bool operator != (const CFixedPoint64 &other) const
	{
		return !operator==(other);
	}

	__device__ inline CFixedPoint64 Sqr();

	__device__ __host__ inline bool IsNeg() const { return m_val64 < 0; }

	__device__ __host__ inline void MakeAbs() { m_val64 = abs(m_val64); }

	__device__ __host__ inline bool IsAbsLargerThan2() const;

	__device__ __host__ inline bool IsOverflown4() const { return IsNeg(); }

};

inline CFixedPoint64::operator float() const
{
	int64_t x = m_val64;
	const bool bNeg = x<0;
	if (bNeg)
		x = -x;
	int lz; // leading zeros
#ifndef __CUDA_ARCH__
	lz = (int)__lzcnt64(x);
#else
	lz = __clzll(x);
#endif
	if (lz == 64)
		return 0;
	int shift = (63 - lz - 23);
	if (shift >= 0)
	{
		x >>= shift;
	}
	else
	{
		x <<= -shift;
	}

	unsigned char e = (127 - lz + 2);

	uint32_t uiResult = e << 23;
	uiResult |= x & 0x007fffff; //cut off the msb (because float always begins with 1).
	if (bNeg)
		uiResult |= 0x80000000;

	return Bin32ToFloat(uiResult);
}

inline CFixedPoint64::CFixedPoint64(const float d)
{
	// from wikipedia:
	// (-1) ** b31  *  ( 1.b22b21b20...b0)_2  *  2 ** ((b30b29...b23)_2 - 127)
	const uint32_t x = FloatToBin32(d);
	uint32_t e = (x & 0x7f800000) >> 23;
	uint32_t f = x & 0x007fffff;
	uint32_t s = x >> 31;
	m_lo = f | 0x00800000;
	m_hi = 0;
	int shft = (int)e + (-127 + 61 - 23);
	if (shft > 0)
		operator <<=(shft);
	else if (shft < 0)
		operator >>=(-shft);

	if (s)
		m_val64 = -m_val64;
}

inline __host__ __device__ CFixedPoint64 CFixedPoint64::operator*(const CFixedPoint64 & other)
{
#ifndef __CUDA_ARCH__
	__int64 hi64;
	__int64 lo64 = _mul128(m_val64, other.m_val64, &hi64);
#else
	int64_t hi64 = __mul64hi(m_val64, other.m_val64);
	int64_t lo64 = m_val64 * other.m_val64;
#endif

	hi64 <<= 3;
	hi64 |= ((uint64_t)lo64) >> 61;
	return CFixedPoint64(hi64);
}

inline __device__ CFixedPoint64 CFixedPoint64::Sqr()
{
	return (*this)*(*this);
}

__device__ __host__ bool CFixedPoint64::IsAbsLargerThan2() const
{
	return ((m_hi & 0x80000000) >> 1) != (m_hi & 0x40000000);
}

inline std::ostream& operator << (std::ostream& o, const CFixedPoint64& x)
{
	std::ios state(0);
	state.copyfmt(o);

	o << (float)x << " {" << std::setfill('0') << std::setw(16) << std::hex << x.m_val64 << "}";
	o.copyfmt(state);
	return o;
}
