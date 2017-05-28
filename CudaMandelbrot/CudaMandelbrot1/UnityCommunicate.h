#pragma once

#include "ExportFunctions.h"
#include "common.h"

LIBRARY_API void __stdcall FillTexture(int nTexNum);

//LIBRARY_API void* __stdcall GetTexture();

LIBRARY_API void __stdcall SetTexture(void* pTex, int nTexNum);

LIBRARY_API void __stdcall MakeCalculation( VEC4_ARG(quatCamConj_list,float) );
