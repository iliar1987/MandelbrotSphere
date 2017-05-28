#pragma once


#define ReactToError(sErr)\
{\
	MessageBoxA(0, sErr, "Error " __FUNCTION__ , 0);\
}

#define ReactToCudaError(err)\
{\
	MessageBoxA(0, cudaGetErrorString(err), "Cuda Error " __FUNCTION__, 0);\
}

#define VEC4_ARG(v,type) type v ## 0, type v ## 1, type v ## 2, type v ## 3
#define VEC4_LIST(v) v ## 0, v ## 1, v ## 2 , v ## 3
#define VEC4_LIST_CONJ(v) v ## 0, -v ## 1, -v ## 2 , -v ## 3

#define PIf 3.14159265f