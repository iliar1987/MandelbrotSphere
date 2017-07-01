#include "MandelbrotKernel.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>
#include <Windows.h>
#include "common.h"

#include "utilities.cuh"

#include "../HighPrecision1/FP128.cuh"

#define OUT
#define IN

typedef void(Func_GetThetaPhi)(OUT float&, OUT float&, int, int, const CTextureFiller::KernelParameters&);

template<Func_GetThetaPhi GetThetaPhi>
__global__ void kernMandelbrot(float* buffer, CTextureFiller::KernelParameters params,CFixedPoint128 xPole,CFixedPoint128 yPole)
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

	float R = params.tFrameParams.rho * tanf( theta / 2.0f) * 2.0f;
	float fX = R*cosf(phi);
	float fY = R*sinf(phi);
	//*pixel = sqrtf(fY*fY + fX*fX);
	//*pixel = theta / PIf;
	//*pixel = powf(sin(theta), 3) * (cosf(3.0f*(phi + params.tFrameParams.t))) * 0.5f + 0.5f;


	if (fX*fX + fY*fY >= 4.0f)
	{
		*pixel = 0.0f;
	}
	else
	{
		CComplexFP128 c(fX, fY);
		c.x += xPole;
		c.y += yPole;
		CComplexFP128 z(c);
		int i = 0;

		while (i < params.tFrameParams.nIterations
			&& !((z.x.hihi & 0x80000000) >> 1 != z.x.hihi & 0x40000000)
			&& !((z.y.hihi & 0x80000000) >> 1 != z.y.hihi & 0x40000000) )
		{
			CFixedPoint128 z_x_sqr = z.x.Sqr();
			CFixedPoint128 z_y_sqr = z.y.Sqr();

			CFixedPoint128 sumOfSquares(z_x_sqr);
			sumOfSquares += z_y_sqr;
			if (sumOfSquares.IsNeg())
				break;

			z.y = z.x * z.y;
			z.y <<= 1;

			z.x = z_x_sqr;
			z_y_sqr.Negate();
			z.x = z_y_sqr;

			z += c;
			++i;
		}
		*pixel = (float)i;

		//z = z.Sqr();
		
		//float fa = (float)(z.x);
		//float fb = (float)(z.y);
		////*pixel = fa*fa + fb*fb;
		////z.Sqr();
		//*pixel = atan2f((float)z.y, (float)z.x);
	}
	
}

void CMandelbrotTextureFiller::LaunchKernel(const KernelParameters& params)
{
	dim3 Db = dim3(8, 8);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((params.width + Db.x - 1) / Db.x, (params.height + Db.y - 1) / Db.y);

	kernMandelbrot<GetThetaPhiSpherical> <<< Dg, Db >>> (GetBuffer(), params,*m_poleCoords.x,*m_poleCoords.y);
}

CMandelbrotTextureFiller::CMandelbrotTextureFiller(int width, int height, float FOV)
	: CTextureFiller(width, height, FOV)
{
	m_poleCoords.x = new CFixedPoint128 { 0,0 };
	m_poleCoords.y = new CFixedPoint128{ 0,0 };
}

CMandelbrotTextureFiller::~CMandelbrotTextureFiller()
{
	delete m_poleCoords.x;
	delete m_poleCoords.y;
}


void CMandelbrotTextureFiller::PoleCoordsGet(float& x, float &y)
{
	x = static_cast<float>(*m_poleCoords.x);
	y = static_cast<float>(*m_poleCoords.y);
}

void CMandelbrotTextureFiller::PoleCoordsAdd(float dx, float dy)
{
	*m_poleCoords.x += CFixedPoint128(dx);
	*m_poleCoords.y += CFixedPoint128(dy);
}

void CMandelbrotTextureFiller::PoleCoordsSet(float x, float y)
{
	*m_poleCoords.x = CFixedPoint128(x);
	*m_poleCoords.y = CFixedPoint128(y);
}

void CMandelbrotTextureFiller::PoleCoordsZoom(float3 vForward, float rho, float rho_new)
{
	float temp = sqrtf(vForward.x * vForward.x + vForward.y * vForward.y);
	float theta = atan2f(temp, vForward.z);
	float phi = atan2f(vForward.y, vForward.x);

	float rho_delta = rho_new - rho;

	float dr = 2 * rho_delta * tanf(theta / 2);
	float dx = dr * cosf(phi);
	float dy = dr * sinf(phi);

	*m_poleCoords.x += CFixedPoint128(-dx);
	*m_poleCoords.y += CFixedPoint128(-dy);
}

