#pragma once

#include "ExportFunctions.h"


LIBRARY_API int __stdcall func1(int x, int y);

LIBRARY_API void __stdcall FillTexture();

LIBRARY_API void* __stdcall GetTexture(int texNum);