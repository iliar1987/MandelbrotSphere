#pragma once

#include <basetsd.h>
#include <stdexcept>

struct ID3D11ShaderResourceView;
struct ID3D11Texture2D;
struct cudaGraphicsResource;
struct ID3D11Device;

class ErrTextureCreate : public std::runtime_error
{
public:
	ErrTextureCreate(const char* s)
		: std::runtime_error(s) {}
};

class ErrTextureUpdate : public std::runtime_error
{
public:
	ErrTextureUpdate(const char* s)
		: std::runtime_error(s) {}
};

class CTextureInfo
{
private:
	const int m_elemSize;

	INT32 m_width;
	INT32 m_height;

	ID3D11Device* m_pDevice;

	ID3D11Texture2D *m_pTex2d;
	//ID3D11ShaderResourceView *m_pSRView;
	cudaGraphicsResource    *m_cudaResource;

	void CreateResource();
	void CreateTexture();

	bool m_bTextureIsMine = false;
public:
	INT32 GetWidth() const { return m_width; }
	INT32 GetHeight() const { return m_height; }

	//ID3D11ShaderResourceView * GetResourceView() { return m_pSRView; }
	ID3D11Texture2D * GetTexture2D() { return  m_pTex2d; }
	cudaGraphicsResource    * GetCudaResource() {return m_cudaResource;}

	CTextureInfo(INT32 width, INT32 height, ID3D11Device* pDevice, const int elemSize);
	CTextureInfo(INT32 width, INT32 height, ID3D11Device* pDevice, ID3D11Texture2D* pTex, const int elemSize);

	CTextureInfo(const CTextureInfo&) = delete;
	CTextureInfo& operator = (const CTextureInfo&) = delete;
	~CTextureInfo();

	void UpdateFromDeviceBuffer(void* d_buffer, size_t pitch);

	CTextureInfo* CreateAnother();
};