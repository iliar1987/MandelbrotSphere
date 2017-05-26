#pragma once

#ifdef LIBRARY_EXPORTS
#	define LIBRARY_API extern "C" __declspec(dllexport)
#else
#	define LIBRARY_API extern "C" __declspec(dllimport)
#endif
