#pragma once

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <assert.h>
#include <intrin.h>

#include "CComplex.cuh"

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

//#define ALL_ONES64 0xffffffffffffffffL

class CFixedPoint128
{
public:
	//static const int s_shiftLeftAfterMul = 3;
	//static const int s_max = 4;
	//static const int s_min = -4;
	static const uint64_t s_allOnes = 0xffffffffffffffffL;
	union
	{
		struct {
			uint64_t lo, hi;
		};
		struct
		{
			uint32_t lolo, lohi, hilo, hihi;
		};
	};

	__device__ __host__ CFixedPoint128() {}
	__device__ __host__ CFixedPoint128(uint64_t _lo, uint64_t _hi)
		: lo(_lo), hi(_hi)
	{}
	__device__ __host__ CFixedPoint128(std::pair<uint64_t, uint64_t> p)
		: lo(p.first), hi(p.second)
	{}

	__host__ __device__ explicit operator float() const;
	__host__ __device__ explicit CFixedPoint128(const float d);

	//__device__ inline CFixedPoint128 operator * (const CFixedPoint128& other) const; //int128 multiplication with multiplication by 8 afterward (to keep place of point).
	__device__ inline CFixedPoint128 operator * (CFixedPoint128 &other);
	__device__ __host__ inline CFixedPoint128 & operator += (const CFixedPoint128 &other);
	__device__ inline CFixedPoint128 & operator -= (const CFixedPoint128 &other);

	__device__ __host__ CFixedPoint128 & operator <<= (const unsigned int n); //in the sense of multiply by power of 2
	__device__ __host__ CFixedPoint128 & operator >>= (const unsigned int n); //in the sense of divide by power of 2

	__device__ inline CFixedPoint128 Sqr();

	__device__ __host__ void Negate(); //switch sign (2's complement)

	__device__ __host__ inline bool IsNeg() const
	{
		return (hihi >> 31) != 0;
	}

};


inline void output(std::ostream &o, const uint64_t x)
{
	o << std::setfill('0') << std::setw(16) << std::hex << x;
}

inline std::ostream& operator << (std::ostream &o, const CFixedPoint128 &x)
{
	o << (float)x << " ";
	o << "{";
	output(o, x.lo);
	o << ", ";
	output(o, x.hi);
	o << "}";
	return o;
}

class CComplexFP128 : public CComplex<CFixedPoint128>
{
public:
	/*__device__ __host__ CComplexFP128(const CComplex<CFixedPoint128>& other)
		: CComplex(other)
	{}*/
	using CComplex::CComplex;
	using CComplex::operator=;
	//CComplexFP128() : CComplex() {}
	__device__ __host__ bool OutsideRadius2() const
	{
//		if ( (((x.hihi & 0x80000000) >> 1) ^ (x.hihi & 0x40000000))
//			|| (((y.hihi & 0x80000000) >> 1) ^ (y.hihi & 0x40000000)) ) //if |x| >= 2 || |y| >= 2
//		{
//			return true;
//		}
//		uint32_t sqr;
//#ifndef __CUDA_ARCH__
//		sqr = static_cast<uint32_t>(((__emul(x.hihi, x.hihi) + __emul(y.hihi, y.hihi))) >> 32);
//#else
//		sqr = __mulhi(x.hihi, x.hihi) + __mulhi(y.hihi,y.hihi);
//#endif
//		return (sqr >> (32 - 3 - 1)) != 0;
		float fX = (float)x;
		float fY = (float)y;
		return fX*fX + fY*fY >= 4.0f;
	}
};

__host__ __device__ inline CFixedPoint128::operator float() const
{
	CFixedPoint128 x(*this);
	const bool bNeg = IsNeg();
	if (bNeg)
		x.Negate();
	int lz; // leading zeros
#ifndef __CUDA_ARCH__
	lz = (int)__lzcnt64(x.hi);
	if (lz == 64)
	{
		lz += (int)__lzcnt64(x.lo);
	}
#else
	lz = __clzll(x.hi);
	if (lz == 64)
	{
		lz += __clzll(x.lo);
	}
#endif
	if (lz == 128)
		return 0;
	x >>= (127 - lz - 23);
	unsigned char e = (127 - lz + 2);

	uint32_t uiResult = e << 23;
	uiResult |= x.lolo & 0x007fffff; //cut off the msb (because float always begins with 1).
	if (bNeg)
		uiResult |= 0x80000000;

	return Bin32ToFloat(uiResult);
}
__host__ __device__ inline CFixedPoint128::CFixedPoint128(const float d)
{
	/*assert(d < 4 && d >= -4);*/
	// from wikipedia:
	// (-1) ** b31  *  ( 1.b22b21b20...b0)_2  *  2 ** ((b30b29...b23)_2 - 127)
	const uint32_t x = FloatToBin32(d);
	uint32_t e = (x & 0x7f800000) >> 23;
	uint32_t f = x & 0x007fffff;
	uint32_t s = x >> 31;
	lolo = f | 0x00800000;
	lohi = 0;
	hi = 0;
	int shft = (int)e + (-127 + 125 - 23);
	if (shft > 0)
		operator <<=(shft);
	else if (shft < 0)
		operator >>=(shft);

	if (s)
		Negate();
	//#ifndef __CUDA_ARCH__
	//		printf("%f = %d\t%d\t%d\n", d, f, e - 127, s);
	//#endif
}


__device__ inline CFixedPoint128 CFixedPoint128::operator * (CFixedPoint128& other)
{
	bool b2IsNeg = other.IsNeg();
	bool b1IsNeg = IsNeg();
	if (b2IsNeg)
	{
		other.Negate();
#ifdef _DEBUG
		if (this == &other) //to avoid bad behavior
			asm("brkpt;");
#endif
	}
	if (b1IsNeg)
		Negate();
	CFixedPoint128 result;
	uint64_t p1;
	asm("{\n\t"
		"mul.hi.u64		%2,	%3,	%5		;\n\t" // p1 = (a0 * b0)_hi
		"mad.lo.cc.u64		%2,	%3,	%6,	%2	;\n\t" // p1 += (a0 * b1)_lo -> C
		"madc.hi.u64     %0, %3, %6,  0		;\n\t" // p2 = (a0 * b1)_hi + C
		"mad.lo.cc.u64		%2,	%4,	%5,	%2	;\n\t" // p1 += (a1 * b0)_lo -> C
		"madc.hi.cc.u64  %0, %4,  %5, %0;\n\t"  // p2 += (a1 * b0)_hi + C -> C
		"madc.hi.u64     %1, %4, %6,  0;\n\t"  // p3 = (a1 * b1)_hi + C
		"mad.lo.cc.u64   %0, %4, %6, %0;\n\t"  // p2 += (a1 * b1)_lo -> C 
		"addc.u64		 %1, %1,	0	;\n\t"  // p3 += C
		"}"
		: "=l"(result.lo), "=l"(result.hi), "=l"(p1)
		: "l"(lo), "l"(hi), "l"(other.lo), "l"(other.hi));

	result<<= (3); // multiply by 8
	result.lo |= (p1 >> 61); //add lower 3 bits from p1.

	if (b2IsNeg)
		other.Negate();
	if (b1IsNeg)
		Negate();

	if (b1IsNeg != b2IsNeg)
		result.Negate();
	//return *this;
	return result;
}

__device__ inline CFixedPoint128 CFixedPoint128::Sqr()
{
	//TODO: make something more efficient
	bool bNeg = IsNeg();
	if (bNeg)
	{
		Negate();
	}
	CFixedPoint128 result = (*this) * (*this);
	if (bNeg)
		Negate();
	return result;
}


__device__ __host__ inline CFixedPoint128 & CFixedPoint128::operator += (const CFixedPoint128 &other)
{
#ifdef __CUDA_ARCH__
	asm("{\n\t"
		"add.cc.u64 %0, %0, %2 ;\n\t"
		"addc.u64 %1, %1, %3   ;\n\t"
		"}" : "+l"(lo), "+l"(hi) : "l"(other.lo), "l"(other.hi));
#else
	unsigned char c = _addcarry_u64(0, lo, other.lo, &lo);
	_addcarry_u64(c, hi, other.hi, &hi);
#endif
	return *this;
}

__device__ inline CFixedPoint128 & CFixedPoint128::operator -= (const CFixedPoint128 &other)
{
	asm("{\n\t"
		"sub.cc.u64 %0, %0, %2 ;\n\t"
		"subc.u64 %1, %1, %3   ;\n\t"
		"}" : "+l"(lo), "+l"(hi) : "l"(other.lo), "l"(other.hi));
	return *this;
}

__device__ __host__ inline CFixedPoint128 & CFixedPoint128::operator <<= (const unsigned int n)
{
	if (n >= 64)
	{
		hi = lo;
		lo = 0;
		hi <<= (n - 64);
	}
	else
	{
		hi <<= n;
		const uint64_t mask = s_allOnes << (64 - n);
		const uint64_t carry = (lo & mask);
		lo <<= n;
		hi |= (carry >> (64 - n));
	}
	return *this;
}

__device__ __host__ inline CFixedPoint128 & CFixedPoint128::operator >>= (const unsigned int n)
{
	const bool bNeg = IsNeg();
	if (n >= 64)
	{
		lo = hi;
		hi = bNeg ? s_allOnes : 0;
		lo >>= (n - 64);
		if (bNeg)
			lo |= (s_allOnes << (128 - n));
	}
	else
	{
		lo >>= n;
		const uint64_t mask = s_allOnes >> (64 - n);
		const uint64_t carry = (hi & mask);
		hi >>= n;
		lo |= (carry << (64 - n)); //TODO: is it n or 2n?
		if (bNeg)
		{
			hi |= (s_allOnes << (64 - n));
		}
	}

	return *this;
}

__device__ __host__ inline void CFixedPoint128::Negate()
{
#ifndef __CUDA_ARCH__
	hi = ~hi;
	lo = ~lo;
	int msb_lo = lohi & 0x80000000;
	lo++;
	if ((!(lohi & 0x80000000)) && msb_lo)
	{
		hi++;
	}
#else
	asm("{\n\t"
		"not.b64 %0, %0;\n\t"
		"not.b64 %1, %1;\n\t"
		"add.cc.u64 %0, %0, 1 ; \n\t"
		"addc.u64 %1, %1, 0; \n\t"
		"}" : "+l"(lo), "+l"(hi));
#endif
}
