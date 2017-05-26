#include "main.h"

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

#include "kernel.h"

#include <stdio.h>
#include <io.h>
#include <fcntl.h>

int __stdcall func1(int x, int y)
{
	return x + y;
}

const int widthTotal = 4096;
const int heightTotal = 4096;

SimpleFillTexture* g_pSimpleFillTexture=nullptr;

void CreateConsole()
{
	BOOL bRes = AllocConsole();

	HANDLE handle_out = GetStdHandle(STD_OUTPUT_HANDLE);
	int hCrt = _open_osfhandle((intptr_t)handle_out, _O_TEXT);
	FILE* hf_out = _fdopen(hCrt, "w");
	setvbuf(hf_out, NULL, _IONBF, 1);
	*stdout = *hf_out;

	HANDLE handle_in = GetStdHandle(STD_INPUT_HANDLE);
	hCrt = _open_osfhandle((intptr_t)handle_in, _O_TEXT);
	FILE* hf_in = _fdopen(hCrt, "r");
	setvbuf(hf_in, NULL, _IONBF, 128);
	*stdin = *hf_in;

	HANDLE handle_err = GetStdHandle(STD_ERROR_HANDLE);
	hCrt = _open_osfhandle((intptr_t)handle_err, _O_TEXT);
	FILE* hf_err = _fdopen(hCrt, "w");
	setvbuf(hf_err, NULL, _IONBF, 1);
	*stderr = *hf_err;

	printf("Hello world\r\n");

	// use the console just like a normal one - printf(), getchar(), ...
}

bool __stdcall Init()
{
	CreateConsole();
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return false;
	}

	g_pSimpleFillTexture = new SimpleFillTexture(widthTotal,heightTotal);

	return true;
}

void __stdcall Cleanup()
{
	if (g_pSimpleFillTexture)
	{
		delete g_pSimpleFillTexture;
		g_pSimpleFillTexture = nullptr;
	}
}

bool __stdcall FillTexture(void* _pTex)
{
	ID3D11Texture2D* pTex = (ID3D11Texture2D*)_pTex;

	cudaGraphicsResource    *cudaResource;
	cudaError_t status;

	status = cudaGraphicsD3D11RegisterResource(&cudaResource, pTex, cudaGraphicsRegisterFlagsNone);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Error in register resource");
		return false;
	}

	g_pSimpleFillTexture->UpdateBuffer();
	float4* buffer = g_pSimpleFillTexture->GetCurrentBuffer();

	cudaArray *cuArray;

	status = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Error in get mapped array");
		return false;
	}

	status = cudaMemcpy2DToArray(cuArray, 0, 0, (void*)buffer, g_pSimpleFillTexture->GetPitch(), g_pSimpleFillTexture->GetWidth(), g_pSimpleFillTexture->GetHeight(), cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Error in cuda copy");
		return false;
	}

	status = cudaGraphicsUnregisterResource(cudaResource);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Error in unregister resource");
		return false;
	}

	return true;
}
