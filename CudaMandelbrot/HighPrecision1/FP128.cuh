#pragma once


typedef unsigned long long int uint64_t;
typedef signed long long int int64_t;

__device__ __forceinline__ void mul_128_128_HI_Approx(const uint64_t a0, const uint64_t a1,

	const uint64_t b0, const uint64_t b1,

	uint64_t &p0, uint64_t &p1)

{


	asm("{\n\t"

		"mul.hi.u64     %0, %2, %5		;\n\t"  /* (a0 * b1)_hi */

		"mad.hi.cc.u64  %0, %3,  %4, %0;\n\t"  /* (a1 * b0)_hi */

		"madc.hi.u64     %1, %3, %5,  0;\n\t"  /* (a1 * b1)_hi */

		"mad.lo.cc.u64   %0, %3, %5, %0;\n\t"  /* (a1 * b1)_lo */

		"addc.u64		 %1, %1,	0	;\n\t"  /* propagate carry */

		"add.cc.u64		 %0, %0, 1		;\n\t" /*add 1 which is average carry for neglecting the lower multiplications*/

		"addc.u64		%1, %1, 0		;\n\t" /*propagate carry from 'add 1'*/

		"}"

		: "=l"(p0), "=l"(p1)

		: "l"(a0), "l"(a1), "l"(b0), "l"(b1));

}

#define ALL_ONES64 0xffffffffffffffffL

class CFixedPoint128
{
public:
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

	__device__ __forceinline__ CFixedPoint128 operator * (const CFixedPoint128& other) const
	{
		CFixedPoint128 result;
		mul_128_128_HI_Approx(lo, hi, other.lo, other.hi, result.lo, result.hi);
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

	__device__ __forceinline__ void Negate()
	{
		asm("{\n\t"
			"not.b64 %0, %0;\n\t"
			"not.b64 %1, %1;\n\t"
			"add.cc.u64 %0, %0, 1 ; \n\t"
			"addc.u64 %1, %1, 0; \n\t"
			"}" : "+l"(lo), "+l"(hi));
	}

	__device__ __forceinline__ void ShiftLeft1()
	{
		asm("{\n\t"
			".reg .b64 temp		;\n\t"
			"shl.b64 %1, %1, 1	;\n\t"
			"and.b64 temp, %0, 0x1000000000000000 ;\n\t"
			"shr.b64 temp, temp, 63 ;\n\t"
			"or.b64 %1,%1,temp	;\n\t"
			"shl.b64 %0, %0, 1	;\n\t"
			"}" : "+l"(lo), "+l"(hi));
	}

	__device__ __forceinline__ void ShiftRight1()
	{
		asm("{\n\t"
			".reg .b64 temp		;\n\t"
			"shr.b64 %0, %0, 1	;\n\t"
			"and.b64 temp, %1, 1 ;\n\t"
			"shl.b64 temp, temp, 63 ;\n\t"
			"or.b64 %0,%0,temp	;\n\t"
			"shr.b64 %1, %1, 1	;\n\t"
			"}" : "+l"(lo), "+l"(hi));
	}

	__host__ __forceinline__ explicit operator double() const
	{
		return 0.0;
	}

	__host__ CFixedPoint128(double d)
	{

	}

	__device__ __host__ __forceinline__ bool IsNeg() const
	{
		return hihi >> 31 ? true : false;
	}


	// TODO: it's possible
	//__device__ __forceinline__ CFixedPoint128 & operator <<= (int n)
	//{
	//	asm("{\n\t"
	//		".reg .b64 temp		;\n\t"
	//		"shl.b64 %1, %1, 1	;\n\t"
	//		"and.b64 temp, %0, 0x1000000000000000 ;\n\t"
	//		"shr.b64 temp, temp, 63 ;\n\t"
	//		"or.b64 %1,%1,temp	;\n\t"
	//		"shl.b64 %0, %0, 1	;\n\t"
	//		"}" : "+l"(lo), "+l"(hi));
	//}

	//__device__ __forceinline__ CFixedPoint128 & operator >>= (int n)
	//{
	//	int complement = 64 - n;
	//	asm("{\n\t"
	//		".reg .b64 temp		;\n\t"
	//		"shr.b64 %0, %0, 1	;\n\t"
	//		"and.b64 temp, %1, 1 ;\n\t"
	//		"shl.b64 temp, temp, %2 ;\n\t"
	//		"or.b64 %0,%0,temp	;\n\t"
	//		"shr.b64 %1, %1, 1	;\n\t"
	//		"}" : "+l"(lo), "+l"(hi));
	//}
};