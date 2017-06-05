#pragma once


#define ReactToError(sErr)\
{\
	char s[512];\
	sprintf_s(s,512,__FILE__ " %d : %s",__LINE__,sErr);\
	MessageBoxA(0, s, "Error " , 0);\
}

#define ReactToCudaError(err)\
{\
	char s[512];\
	sprintf_s(s,512,__FILE__ " %d : %s",__LINE__,cudaGetErrorString(err));\
	MessageBoxA(0, s, "Cuda Error ", 0);\
}

#define VEC4_ARG(v,type) type v ## 0, type v ## 1, type v ## 2, type v ## 3
#define VEC4_LIST(v) v ## 0, v ## 1, v ## 2 , v ## 3
#define VEC4_LIST_CONJ(v) v ## 0, -v ## 1, -v ## 2 , -v ## 3

#define PIf 3.14159265f


