#pragma once

#include <intrin.h>
#include <iostream>

#include "CommonHP.cuh"

class CFixedPoint32
{
public:
	int32_t m_val32;

	__device__ __host__ CFixedPoint32() {}
	__device__ __host__ explicit CFixedPoint32(int32_t v32)
		: m_val32(v32)
	{}

	__host__ __device__ inline explicit operator float() const;
	__host__ __device__ inline explicit CFixedPoint32(const float d);

	__host__ __device__ inline CFixedPoint32 operator * (const CFixedPoint32 &other);

	__device__ __host__ inline CFixedPoint32 & operator += (const CFixedPoint32 &other)
	{
		m_val32 += other.m_val32;
		return *this;
	}

	__device__ __host__ inline CFixedPoint32 & operator -= (const CFixedPoint32 &other)
	{
		m_val32 -= other.m_val32;
		return *this;
	};

	__device__ __host__ inline CFixedPoint32 & operator <<= (const unsigned char n) { m_val32 <<= n; return *this; }
	__device__ __host__ inline CFixedPoint32 & operator >>= (const unsigned char n) { m_val32 >>= n; return *this; }

	__device__ __host__ bool operator == (const CFixedPoint32& other) const
	{
		return m_val32 == other.m_val32;
	}

	__device__ __host__ bool operator != (const CFixedPoint32 &other) const
	{
		return !operator==(other);
	}

	__device__ inline CFixedPoint32 Sqr();

	__device__ __host__ inline bool IsNeg() const { return m_val32 < 0; }

	__device__ __host__ inline void MakeAbs() { m_val32 = abs(m_val32); }

	__device__ __host__ inline bool IsAbsLargerThan2() const;

	__device__ __host__ inline bool IsOverflown4() const { return IsNeg(); }

};

inline CFixedPoint32::operator float() const
{
	int32_t x = m_val32;
	const bool bNeg = x<0;
	if (bNeg)
		x = -x;
	int lz; // leading zeros
#ifndef __CUDA_ARCH__
	lz = (int)__lzcnt(x);
#else
	lz = __clz(x);
#endif
	if (lz == 32)
		return 0;
	int shift = (31 - lz - 23);
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

inline CFixedPoint32::CFixedPoint32(const float d)
{
	// from wikipedia:
	// (-1) ** b31  *  ( 1.b22b21b20...b0)_2  *  2 ** ((b30b29...b23)_2 - 127)
	const uint32_t x = FloatToBin32(d);
	uint32_t e = (x & 0x7f800000) >> 23;
	uint32_t f = x & 0x007fffff;
	uint32_t s = x >> 31;
	m_val32 = f | 0x00800000;
	int shft = (int)e + (-127 + 29 - 23);
	if (shft > 0)
		operator <<=(shft);
	else if (shft < 0)
		operator >>=(-shft);

	if (s)
		m_val32 = -m_val32;
}

inline __host__ __device__ CFixedPoint32 CFixedPoint32::operator*(const CFixedPoint32 & other)
{
	int32_t hi32;
#ifndef __CUDA_ARCH__
	int64_t x64 = __emul(m_val32, other.m_val32);
	hi32 = (int32_t) (x64 >> (32-3));
#else
	hi32 = __mulhi(m_val32, other.m_val32);
	int32_t lo32 = m_val32 * other.m_val32;
	hi32 <<= 3;
	hi32 |= ((uint32_t)lo32) >> 29;
#endif

	return CFixedPoint32(hi32);
}

inline __device__ CFixedPoint32 CFixedPoint32::Sqr()
{
	return (*this)*(*this);
}

__device__ __host__ bool CFixedPoint32::IsAbsLargerThan2() const
{
	return ((m_val32 & 0x80000000) >> 1) != (m_val32 & 0x40000000);
}

inline std::ostream& operator << (std::ostream& o, const CFixedPoint32& x)
{
	std::ios state(0);
	state.copyfmt(o);

	o << (float)x << " {" << std::setfill('0') << std::setw(8) << std::hex << x.m_val32 << "}";
	o.copyfmt(state);
	return o;
}
