
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <utility>

#include <iostream>

#include "FP128.cuh"

__global__ void mulKernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}


__global__ void shlKernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
	int i = threadIdx.x;
	CFixedPoint128 temp = a[i];
	temp <<= 1;
	c[i] = temp;
}

__global__ void shrKernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
	int i = threadIdx.x;
	CFixedPoint128 temp = a[i];
	temp >>= 1;
	c[i] = temp;
}

__global__ void divideByPow2Kernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
	int i = threadIdx.x;
	CFixedPoint128 temp = a[i];
	temp >>= (i+5);
	c[i] = temp;
}


__global__ void isNegKernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
	int i = threadIdx.x;
	c[i].hi = 0;
	if (a[i].IsNeg())
	{
		c[i].lo = 1;
	}
	else
	{
		c[i].lo = 0;
	}
}

__global__ void subtractKernel(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b)
{
	int i = threadIdx.x;

	/*CFixedPoint128 x = a[i];
	x.Negate();
	x += b[i];
	c[i] = x;*/
	CFixedPoint128 x = b[i];
	x -= a[i];
	c[i] = x;
}

void PrintFromDouble(float f)
{
	CFixedPoint128 fp128(f);
	uint32_t& f_x = *(reinterpret_cast<uint32_t*>(&f));
	printf("%08x = ", f_x);
	std::cout << fp128 << std::endl;
}

void TestFromDouble()
{
	float arr[] = { 0.25f,0.5f,1.0f,1.5f,1.25f,0.75f,-1.0f,2.0f,4.0f };
	for (float x : arr)
	{
		PrintFromDouble(x);
	}
}

typedef void CudaOp(CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b);

cudaError_t PerformOpWithCuda(CudaOp* op, CFixedPoint128 *c, const CFixedPoint128 *a, const CFixedPoint128 *b, unsigned int size);

int main()
{
	TestFromDouble();


    const int arraySize = 3;
	const CFixedPoint128 a[arraySize] = { {0x1010101010101010L,0x1010101010101010L },{ 0x2020202020202020L,0x4020202020202020L },{ 0,0x2000000000000000L } };
	const CFixedPoint128 b[arraySize] = { { 0x3010101010101010L, 0x1010101010101010L }, { 0x2020202020202020L, 0x2020202020202020L },{ 0,0x2000000000000000L } };
	CFixedPoint128 c[arraySize] = { {0,0},{1,1} };
	cudaError_t cudaStatus;

    cudaStatus = PerformOpWithCuda(&mulKernel, c, a, b, arraySize);
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

	memcpy((void*)a, (void*)c, arraySize * sizeof(CFixedPoint128));
	cudaStatus = PerformOpWithCuda(&isNegKernel, c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "isneg failed!");
		return 1;
	}
	for (int i = 0;i < arraySize;++i)
	{
		std::cout << a[i] << ".IsNeg() \t=\t" << c[i] << std::endl;
	}
	
	cudaStatus = PerformOpWithCuda(&shlKernel, c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "shl failed!");
		return 1;
	}
	for (int i = 0;i < arraySize;++i)
	{
		std::cout << a[i] << "<<1 \t=\t" << c[i] << std::endl;
	}

	cudaStatus = PerformOpWithCuda(&shrKernel, c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "shl failed!");
		return 1;
	}
	for (int i = 0;i < arraySize;++i)
	{
		std::cout << a[i] << ">>1 \t=\t" << c[i] << std::endl;
	}

	cudaStatus = PerformOpWithCuda(&divideByPow2Kernel, c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "divide by pow 2 failed!");
		return 1;
	}
	for (int i = 0;i < arraySize;++i)
	{
		std::cout << a[i] << "/2**"<<i+5<<" \t=\t" << c[i] << std::endl;
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
