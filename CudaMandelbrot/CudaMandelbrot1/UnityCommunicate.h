#pragma once

#include "ExportFunctions.h"
#include "common.h"

LIBRARY_API void __stdcall FillTexture(int nTexNum);

//LIBRARY_API void* __stdcall GetTexture();

LIBRARY_API void __stdcall SetTexture(void* pTex, int nTexNum);

LIBRARY_API void __stdcall MakeCalculation( float vCamRight[3],float vCamUp[3], float vCamForward[3], float t, float rho, int nIterations);

LIBRARY_API void __stdcall Init(bool bDebug, int width, int height, float FOV,const char* fractalName);

LIBRARY_API void __stdcall Shutdown();

LIBRARY_API void __stdcall PoleCoordsGet(float* x, float *y);

LIBRARY_API void __stdcall PoleCoordsAdd(float dx, float dy);

LIBRARY_API void __stdcall PoleCoordsSet(float x, float y);

LIBRARY_API void __stdcall PoleCoordsZoom(float vCamForward[3], float rho,float rho_new);
