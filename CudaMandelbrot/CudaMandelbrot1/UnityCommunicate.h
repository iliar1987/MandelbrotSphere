#pragma once

#include "ExportFunctions.h"

LIBRARY_API void __stdcall FillTexture(int nTexNum);

//LIBRARY_API void* __stdcall GetTexture();

LIBRARY_API void __stdcall SetTexture(void* pTex, int nTexNum);
