
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <utility>

#include <iostream>


typedef unsigned long long int uint64_t;
typedef signed long long int int64_t;

//
//__device__ __forceinline__ void mul_96_192(const  *a,
//
//	const unsigned int *b,
//
//	unsigned int *p)
//
//{
//
//	unsigned int a0 = a[0];
//
//	unsigned int a1 = a[1];
//
//	unsigned int a2 = a[2];
//
//	unsigned int b0 = b[0];
//
//	unsigned int b1 = b[1];
//
//	unsigned int b2 = b[2];
//
//	unsigned int p0, p1, p2, p3, p4, p5;
//
//
//
//	asm("{\n\t"
//
//		"mul.lo.u32      %0, %6,  %9    ;\n\t"  /* (a0 * b0)_lo */
//
//		"mul.hi.u32      %1, %6,  %9    ;\n\t"  /* (a0 * b0)_hi */
//
//		"mad.lo.cc.u32   %1, %6, %10, %1;\n\t"  /* (a0 * b1)_lo */
//
//		"madc.hi.u32     %2, %6, %10,  0;\n\t"  /* (a0 * b1)_hi */
//
//		"mad.lo.cc.u32   %1, %7,  %9, %1;\n\t"  /* (a1 * b0)_lo */
//
//		"madc.hi.cc.u32  %2, %7,  %9, %2;\n\t"  /* (a1 * b0)_hi */
//
//		"madc.hi.u32     %3, %6, %11,  0;\n\t"  /* (a0 * b2)_hi */
//
//		"mad.lo.cc.u32   %2, %6, %11, %2;\n\t"  /* (a0 * b2)_lo */
//
//		"madc.hi.cc.u32  %3, %7, %10, %3;\n\t"  /* (a1 * b1)_hi */
//
//		"madc.hi.u32     %4, %7, %11,  0;\n\t"  /* (a1 * b2)_hi */
//
//		"mad.lo.cc.u32   %2, %7, %10, %2;\n\t"  /* (a1 * b1)_lo */
//
//		"madc.hi.cc.u32  %3, %8,  %9, %3;\n\t"  /* (a2 * b0)_hi */
//
//		"madc.hi.cc.u32  %4, %8, %10, %4;\n\t"  /* (a2 * b1)_hi */
//
//		"madc.hi.u32     %5, %8, %11,  0;\n\t"  /* (a2 * b2)_hi */
//
//		"mad.lo.cc.u32   %2, %8,  %9, %2;\n\t"  /* (a2 * b0)_lo */
//
//		"madc.lo.cc.u32  %3, %7, %11, %3;\n\t"  /* (a1 * b2)_lo */
//
//		"madc.lo.cc.u32  %4, %8, %11, %4;\n\t"  /* (a2 * b2)_lo */
//
//		"addc.u32        %5, %5,   0;    \n\t"  /* propagate carry */
//
//		"mad.lo.cc.u32   %3, %8, %10, %3;\n\t"  /* (a2 * b1)_lo */
//
//		"addc.cc.u32     %4, %4,   0    ;\n\t"  /* propagate carry */
//
//		"addc.u32        %5, %5,   0    ;\n\t"  /* propagate carry */
//
//		"}"
//
//		: "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3), "=r"(p4), "=r"(p5)
//
//		: "r"(a0), "r"(a1), "r"(a2), "r"(b0), "r"(b1), "r"(b2));
//
//
//
//	p[0] = p0;
//
//	p[1] = p1;
//
//	p[2] = p2;
//
//	p[3] = p3;
//
//	p[4] = p4;
//
//	p[5] = p5;
//
//}


//__device__ __forceinline__ void mul_128_256(const uint64_t a[2],
//
//	const uint64_t b[2],
//
//	uint64_t p[4])
//
//{
//
//	unsigned int a0 = a[0];
//
//	unsigned int a1 = a[1];
//
//	unsigned int b0 = b[0];
//
//	unsigned int b1 = b[1];
//
//	unsigned int p0, p1, p2, p3;
//
//
//
//	asm("{\n\t"
//
//		"mul.lo.u32      %0, %4,  %6    ;\n\t"  /* (a0 * b0)_lo */
//
//		"mul.hi.u32      %1, %4,  %6    ;\n\t"  /* (a0 * b0)_hi */
//
//		"mad.lo.cc.u32   %1, %4, %7, %1;\n\t"  /* (a0 * b1)_lo */
//
//		"madc.hi.u32     %2, %4, %7,  0;\n\t"  /* (a0 * b1)_hi */
//
//		"mad.lo.cc.u32   %1, %5,  %6, %1;\n\t"  /* (a1 * b0)_lo */
//
//		"madc.hi.cc.u32  %2, %5,  %6, %2;\n\t"  /* (a1 * b0)_hi */
//
//		"madc.hi.u32     %3, %5, %7,  0;\n\t"  /* (a1 * b1)_hi */
//
//		"mad.lo.cc.u32   %2, %5, %7, %2;\n\t"  /* (a1 * b1)_lo */
//
//		"addc.u32		 %3, %3,	0	;\n\t"  /* propagate carry */
//
//		"}"
//
//		: "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
//
//		: "r"(a0), "r"(a1), "r"(b0), "r"(b1);
//
//
//
//	p[0] = p0;
//
//	p[1] = p1;
//
//	p[2] = p2;
//
//	p[3] = p3;
//
//}

__device__ __forceinline__ void mul_128_128_HI_Approx(const uint64_t a0, const uint64_t a1,

	const uint64_t b0, const uint64_t b1,

	uint64_t &p0, uint64_t &p1)

{

	//unsigned int a0 = a[0];

	//unsigned int a1 = a[1];

	//unsigned int b0 = b[0];

	//unsigned int b1 = b[1];

	//unsigned int p0, p1;



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



	//p[0] = p0;

	//p[1] = p1;

}

#define ALL_ONES64 0xffffffffffffffffL

class CFixedPoint128
{
public:
	uint64_t lo, hi;
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

	__device__ __forceinline__ void Neg() 
	{
		asm("{\n\t"
			"not.b64 %0, %0;\n\t"
			"not.b64 %1, %1;\n\t"
			"add.cc.u64 %0, %0, 1 ; \n\t"
			"addc.u64 %1, %1, 0; \n\t"
			"}" : "+l"(lo), "+l"(hi));
	}
};

std::ostream& operator << (std::ostream &o, const CFixedPoint128 &x)
{
	o << "{" << std::hex << x.lo << ", " << std::hex << x.hi << "}" << std::hex;
	return o;
}


__global__ void mulKernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}


__global__ void subtractKernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
	int i = threadIdx.x;

	/*CFixedPoint128 x = a[i];
	x.Neg();
	x += b[i];
	c[i] = x;*/
	CFixedPoint128 x = b[i];
	x -= a[i];
	c[i] = x;
}



typedef void CudaOp(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b);

cudaError_t PerformOpWithCuda(CudaOp* op, CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b, unsigned int size);

int main()
{
    const int arraySize = 2;
	const CFixedPoint128 a[arraySize] = { {0x1010101010101010L,0x1010101010101010L },{ 0x2020202020202020L,0x4020202020202020L } };
	const CFixedPoint128 b[arraySize] = { { 0x3010101010101010L, 0x1010101010101010L }, { 0x2020202020202020L, 0x2020202020202020L } };
	CFixedPoint128 c[arraySize] = { {0,0},{1,1} };

    // Add vectors in parallel.
    cudaError_t cudaStatus = PerformOpWithCuda(&mulKernel, c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mul failed!");
        return 1;
    }
	for (int i = 0;i < arraySize;++i)
	{

		std::cout << a[i] << "\t*\t" << b[i] << "\t=\t" << c[i] << std::endl;

	}

	cudaStatus = PerformOpWithCuda(&subtractKernel,c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "subtract failed!");
		return 1;
	}
	for (int i = 0;i < arraySize;++i)
	{
		std::cout << b[i] << "\t-\t" << a[i] << "\t=\t" << c[i] << std::endl;
	}
	
    

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t PerformOpWithCuda(CudaOp* op, CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b, unsigned int size)
{
	CFixedPoint128 *dev_a = 0;
	CFixedPoint128 *dev_b = 0;
	CFixedPoint128 *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(CFixedPoint128));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(CFixedPoint128));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(CFixedPoint128));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(CFixedPoint128), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(CFixedPoint128), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    (*op)<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(CFixedPoint128), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
