#pragma once

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <assert.h>

typedef unsigned long long int uint64_t;
typedef signed long long int int64_t;

//#define ALL_ONES64 0xffffffffffffffffL

class CFixedPoint128
{
public:
	static const int s_shiftRightAfterMul = 3;
	static const int s_max = 4;
	static const int s_min = -4;
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
	__device__ __host__ CFixedPoint128(uint64_t lo, uint64_t hi)
		: lo(lo), hi(hi)
	{
	}

	__device__ __host__ CFixedPoint128(std::pair<uint64_t, uint64_t> p)
		: lo(p.first), hi(p.second)
	{}

	__host__ __device__ explicit operator float() const
	{
		return 0.0; //__ffs
	}

	__host__ __device__ explicit CFixedPoint128(const float d)
	{
		/*assert(d < 4 && d >= -4);*/
		// from wikipedia:
		// (-1) ** b31  *  ( 1.b22b21b20...b0)_2  *  2 ** ((b30b29...b23)_2 - 127)
		const uint32_t &x = *(reinterpret_cast<const uint32_t*>(&d));
		uint32_t e = (x & 0x7f800000) >> 23;
		uint32_t f = x & 0x007fffff;
		uint32_t s = x >> 31;
		lolo = f | 0x800000;
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

	__device__ __forceinline__ CFixedPoint128 operator * (const CFixedPoint128& other) const
	{
		CFixedPoint128 result;
		uint64_t p1;
		asm ("{\n\t"
			".reg .u64 %2					;\n\t" //declare p1 register
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

		// multiply by 8 and add from %2
		result <<= 3;
		result.lo |= (p1 << 61); //add lower 3 bits from %2.

		return result;
	}

	__device__ __forceinline__ CFixedPoint128 & operator += (const CFixedPoint128 &other)
	{
		asm("{\n\t"
			"add.cc.u64 %0, %0, %2 ;\n\t"
			"addc.u64 %1, %1, %3   ;\n\t"
			"}" : "+l"(lo), "+l"(hi) : "l"(other.lo), "l"(other.hi));
		return *this;
	}

	__device__ __forceinline__ CFixedPoint128 & operator -= (const CFixedPoint128 &other)
	{
		asm("{\n\t"
			"sub.cc.u64 %0, %0, %2 ;\n\t"
			"subc.u64 %1, %1, %3   ;\n\t"
			"}" : "+l"(lo), "+l"(hi) : "l"(other.lo), "l"(other.hi));
		return *this;
	}

	__device__ __host__ CFixedPoint128 & operator <<= (const unsigned int n)
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

	__device__ __host__ CFixedPoint128 & operator >>= (const unsigned int n)
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

	__device__ __host__ void Negate()
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

	__device__ __host__ __forceinline__ bool IsNeg() const
	{
		return hihi >> 31 ? true : false;
	}

};


void output(std::ostream &o, const uint64_t x)
{
	o << std::setfill('0') << std::setw(16) << std::hex << x;
}

std::ostream& operator << (std::ostream &o, const CFixedPoint128 &x)
{
	o << "{";
	output(o, x.lo);
	o << ", ";
	output(o, x.hi);
	o << "}";
	return o;
}

