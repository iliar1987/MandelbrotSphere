#pragma once
#include "TextureFiller.h"
#include "..\HighPrecision1\FP128.cuh"
#include "utilities.cuh"

class CFixedPoint128;

typedef void(Func_GetThetaPhi)(OUT float&, OUT float&, int, int, const CTextureFiller::KernelParameters&);

typedef bool (Func_Iterate)(IN OUT CComplexFP128 & z, IN const CComplexFP128 & c);

template<Func_Iterate funcIterate>
class CComplexIterationTextureFiller : public CTextureFiller
{
	virtual void LaunchKernel(const KernelParameters& params) override;
public:
	using CTextureFiller::CTextureFiller;

};


template<Func_GetThetaPhi GetThetaPhi, Func_Iterate funcIterate>
__global__ void kernComplexIteration(float* buffer, CTextureFiller::KernelParameters params, CFixedPoint128 xPole, CFixedPoint128 yPole)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	//if (params.tFrameParams.t > 10.0 && x == 1000 && y == 700)
	//{
	//	asm("brkpt;");
	//}

	if (x >= params.width || y >= params.height) return;

	float *pixel = buffer + (y * params.width + x);

	float theta, phi;
	GetThetaPhi(theta, phi, x, y, params);

	float R = params.tFrameParams.rho * tanf(theta / 2.0f) * 2.0f;
	float fX = R*cosf(phi);
	float fY = R*sinf(phi);

	CComplexFP128 c(fX, fY);
	c.x += xPole;
	c.y += yPole;
	CComplexFP128 z(c);
	int i = 0;

	while (i < params.tFrameParams.nIterations && !funcIterate(z, c))
	{
		++i;
	}
	*pixel = (float)i;
}

template<Func_Iterate funcIterate>
void CComplexIterationTextureFiller<funcIterate>::LaunchKernel(const KernelParameters& params)
{
	dim3 Db = dim3(8, 8);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((params.width + Db.x - 1) / Db.x, (params.height + Db.y - 1) / Db.y);

	kernComplexIteration<GetThetaPhiSpherical,funcIterate> << < Dg, Db >> > (GetBuffer(), params, *m_poleCoords.x, *m_poleCoords.y);
}
