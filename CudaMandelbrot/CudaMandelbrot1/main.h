#pragma once

#include <windows.h>

#include <d3d11.h>


#ifdef LIBRARY_EXPORTS
#	define LIBRARY_API extern "C" __declspec(dllexport)
#else
#	define LIBRARY_API extern "C" __declspec(dllimport)
#endif

LIBRARY_API int __stdcall func1(int x, int y);

LIBRARY_API bool __stdcall FillTexture(void* _pTex);


LIBRARY_API bool __stdcall Init();

LIBRARY_API void __stdcall Cleanup();
