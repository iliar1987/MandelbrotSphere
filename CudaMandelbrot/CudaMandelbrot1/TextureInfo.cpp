
#include "TextureInfo.h"

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

#include "common.h"

void CTextureInfo::CreateTexture()
{
	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
	desc.Width = m_width;
	desc.Height = m_height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R32_FLOAT;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	if (FAILED(m_pDevice->CreateTexture2D(&desc, NULL, &m_pTex2d)))
		ReactToError("Error in CreateTexture2D");
	m_bTextureIsMine = true;
}

void CTextureInfo::CreateResource()
{
	/*if (FAILED(m_pDevice->CreateShaderResourceView(m_pTex2d, NULL, &m_pSRView)))
		ReactToError("Error in CreateShaderResourceView");
*/
	cudaError_t err = cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_pTex2d, cudaGraphicsRegisterFlagsNone);
	if (err != cudaSuccess)
	{
		ReactToCudaError(err);
	}
}

CTextureInfo::CTextureInfo(INT32 width, INT32 height, ID3D11Device* pDevice)
	: m_width(width), m_height(height), m_pDevice(pDevice), m_elemSize(sizeof(float))
{
	CreateTexture();
	CreateResource();
}

CTextureInfo::CTextureInfo(INT32 width, INT32 height, ID3D11Device* pDevice, ID3D11Texture2D* pTex, const int elemSize)
	: m_width(width), m_height(height), m_pDevice(pDevice), m_elemSize(elemSize)
{
	m_pTex2d = pTex;

	CreateResource();
}

CTextureInfo::~CTextureInfo()
{
	// unregister the Cuda resources
	cudaError_t err= cudaGraphicsUnregisterResource(m_cudaResource);
	if (err != cudaSuccess)
		ReactToCudaError(err);
	
	//m_pSRView->Release();
	if ( m_bTextureIsMine)
		m_pTex2d->Release();
}

CTextureInfo* CTextureInfo::CreateAnother()
{
	return new CTextureInfo(m_width, m_height,m_pDevice);
}

