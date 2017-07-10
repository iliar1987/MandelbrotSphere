#pragma once

class CTextureInfo;
class CFixedPoint128;

class CTextureFiller
{
private:
	float* m_d_buffer = nullptr;
	const int m_width;
	const int m_height;
	const float m_fL;
	size_t m_pitch;
public:
	struct KernelParameters;
private:
	virtual void LaunchKernel(const KernelParameters& params) = 0;
protected:
	
	float GetL() const { return m_fL; }
	float* GetBuffer() { return m_d_buffer; }
public:
	struct FrameParameters
	{
		float3 vCamRight;
		float3 vCamUp;
		float3 vCamForward;
		float t;
		float rho;
		int nIterations;
	};
	struct KernelParameters
	{
		FrameParameters tFrameParams;
		int width;
		int height;
		size_t pitch;
		float L;
	};
public:

	int GetWidth() const { return m_width; }
	int GetHeight() const { return m_height; }
	size_t GetPitch() const { return m_pitch; }

	CTextureFiller(int width, int height, float FOV);
	CTextureFiller(const CTextureFiller&) = delete;
	CTextureFiller& operator = (const CTextureFiller&) = delete;

	virtual ~CTextureFiller();
	void UpdateBuffer(const FrameParameters &params);

	void CTextureFiller::FillTexture(CTextureInfo& tex);

protected:
	struct TPoleCoords
	{
		CFixedPoint128* x = nullptr;
		CFixedPoint128* y = nullptr;
	} m_poleCoords;

public:

	void PoleCoordsGet(float& x, float &y);

	void PoleCoordsAdd(float dx, float dy);

	void PoleCoordsSet(float x, float y);

	void PoleCoordsZoom(float3 vCamForward, float rho, float rho_new);
};
