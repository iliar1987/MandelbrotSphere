#pragma once

#include "ExportFunctions.h"
#include "common.h"

LIBRARY_API void __stdcall FillTexture(int nTexNum);

//LIBRARY_API void* __stdcall GetTexture();

LIBRARY_API void __stdcall SetTexture(void* pTex, int nTexNum);

LIBRARY_API void __stdcall MakeCalculation( float vCamRight[3],float vCamUp[3], float vCamForward[3], float t, float rho);

LIBRARY_API void __stdcall Init(bool bDebug = false);

LIBRARY_API void __stdcall Shutdown();

