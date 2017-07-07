#pragma once

#include "TextureFiller.h"

class CFixedPoint128;

#define OUT
#define IN

typedef void(Func_GetThetaPhi)(OUT float&, OUT float&, int, int, const CTextureFiller::KernelParameters&);

typedef bool (Func_Iterate)(IN OUT CFixedPoint128 &, IN OUT CFixedPoint128 &);

template<Func_Iterate funcIterate>
class CComplexIterationTextureFiller : public CTextureFiller
{
	virtual void LaunchKernel(const KernelParameters& params) override;
public:
	CComplexIterationTextureFiller(int width, int height, float FOV);
	virtual ~CComplexIterationTextureFiller();

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
