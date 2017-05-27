#pragma once


#define ReactToError(sErr)\
{\
	MessageBoxA(0, sErr, "Error " __FUNCTION__ , 0);\
}

#define ReactToCudaError(err)\
{\
	MessageBoxA(0, cudaGetErrorString(err), "Cuda Error " __FUNCTION__, 0);\
}
