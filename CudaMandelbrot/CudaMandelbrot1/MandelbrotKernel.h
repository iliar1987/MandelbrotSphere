#pragma once

#include "TextureFiller.h"

class CFixedPoint128;

class CMandelbrotTextureFiller : public CTextureFiller
{
	virtual void LaunchKernel(const KernelParameters& params) override;
public:
	CMandelbrotTextureFiller(int width, int height, float FOV);
	virtual ~CMandelbrotTextureFiller();

private:
	struct TPoleCoords
	{
		CFixedPoint128* x=nullptr;
		CFixedPoint128* y=nullptr;
	} m_poleCoords;
	
public:

	void PoleCoordsGet(float& x, float &y);

	void PoleCoordsAdd(float dx, float dy);

	void PoleCoordsSet(float x, float y);

	void PoleCoordsZoom(float3 vCamForward, float rho, float rho_new);

};
