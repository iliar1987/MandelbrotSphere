
#include "TextureInfo.h"

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

#include "helper_cuda.h"

CTextureInfo::CTextureInfo(INT32 width, INT32 height, ID3D11Device* pDevice)
	: m_width(width), m_height(height), m_pDevice(pDevice)
{
	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
	desc.Width = m_width;
	desc.Height = m_height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	if (FAILED(pDevice->CreateTexture2D(&desc, NULL, &m_pTex2d)))
		throw ErrTextureCreate(__FUNCTION__ "Error in CreateTexture2D");

	if (FAILED(pDevice->CreateShaderResourceView(m_pTex2d, NULL, &m_pSRView)))
		throw ErrTextureCreate(__FUNCTION__ "Error in CreateShaderResourceView");

	cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_pTex2d, cudaGraphicsRegisterFlagsNone);
	getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");

}

CTextureInfo::~CTextureInfo()
{
	// unregister the Cuda resources
	cudaGraphicsUnregisterResource(m_cudaResource);
	getLastCudaError("cudaGraphicsUnregisterResource (g_texture_2d) failed");
	m_pSRView->Release();
	m_pTex2d->Release();
}

CTextureInfo* CTextureInfo::CreateAnother()
{
	return new CTextureInfo(m_width, m_height,m_pDevice);
}

void CTextureInfo::UpdateFromDeviceBuffer(float4* d_buffer,size_t pitch)
{
	cudaError_t status;

	cudaArray *cuArray;

	status = cudaGraphicsSubResourceGetMappedArray(&cuArray, m_cudaResource, 0, 0);
	if (status != cudaSuccess)
	{
		throw(ErrTextureUpdate("Error in get mapped array"));
	}

	status = cudaMemcpy2DToArray(cuArray, 0, 0, (void*)d_buffer, pitch, m_width, m_height, cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess)
	{
		throw(ErrTextureUpdate("Error in Memcpy2d"));
	}

}