#include <stdexcept>

#include <Windows.h>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D11.h"

#include "TextureInfo.h"

#include "SimpleFillTexture.h"
#include "MandelbrotKernel.h"
#include "BurningShipKernel.h"

#include "UnityCommunicate.h"

#include <map>

#include "common.h"

/* passing textures to/from unity:
On Direct3D - like devices this returns a pointer to the base texture type(IDirect3DBaseTexture9 on D3D9, ID3D11Resource on D3D11, ID3D12Resource on D3D12).On OpenGL - like devices the GL texture "name" is returned; cast the pointer to integer type to get it.On Metal, the id<MTLTexture> pointer is returned.On platforms that do not support native code plugins, this function always returns NULL.
*/

//IDXGIAdapter           *g_pCudaCapableAdapter = NULL;  // Adapter to use

IUnityInterfaces* g_UnityInterfaces = NULL;
ID3D11Device* g_Device=NULL;
//ID3D11DeviceContext    *g_pd3dDeviceContext = NULL;
IUnityGraphics* g_Graphics = NULL;
UnityGfxRenderer g_RendererType = kUnityGfxRendererNull;

CTextureFiller* g_pSimpleFillTexture = nullptr;

std::map<int, CTextureInfo*> g_mapTextures;


LIBRARY_API void __stdcall Init(bool bDebug,int width,int height,float FOV,const char* fractalName)
{
	//g_pSimpleFillTexture = new SimpleFillTexture(width, height,FOV);
	if( bDebug)
		SetEnvironmentVariableA("NSIGHT_CUDA_DEBUGGER", "1");
	if (strcmpi(fractalName, "mandelbrot") == 0)
		g_pSimpleFillTexture = CreateMandelbrotTextureFiller(width, height, FOV);
	else if (strcmpi(fractalName, "burningship") == 0)
		g_pSimpleFillTexture = CreateBurningShipTextureFiller(width, height, FOV);
	//g_pSimpleFillTexture = new SimpleFillTexture(width, height, FOV);
}

LIBRARY_API void __stdcall Shutdown()
{
	for (auto& x : g_mapTextures)
	{
		delete x.second;
	}
	g_mapTextures.clear();
	
	delete g_pSimpleFillTexture;
	g_pSimpleFillTexture = nullptr;
}

LIBRARY_API void __stdcall FillTexture(int nTexNum)
{
	g_pSimpleFillTexture->FillTexture(* g_mapTextures[nTexNum]);
}

LIBRARY_API void __stdcall MakeCalculation(float vCamRight[3], float vCamUp[3], float vCamForward[3],float t,float rho, int nIterations)
{
	CTextureFiller::FrameParameters params;
	params.t = t;
	params.vCamForward = ARR_AS_FLOAT3(vCamForward);
	params.vCamRight = ARR_AS_FLOAT3(vCamRight);
	params.vCamUp = ARR_AS_FLOAT3(vCamUp);
	params.rho = rho;
	params.nIterations = nIterations;
	g_pSimpleFillTexture->UpdateBuffer(params);
}

LIBRARY_API void __stdcall PoleCoordsGet(float* x, float *y)
{
	g_pSimpleFillTexture->PoleCoordsGet(*x, *y);
}

LIBRARY_API void __stdcall PoleCoordsAdd(float dx, float dy)
{
	g_pSimpleFillTexture->PoleCoordsAdd(dx,dy);
}

LIBRARY_API void __stdcall PoleCoordsSet(float x, float y)
{
	g_pSimpleFillTexture->PoleCoordsSet(x, y);
}

LIBRARY_API void __stdcall PoleCoordsZoom(float vCamForward[3], float rho, float rho_new)
{
	g_pSimpleFillTexture->PoleCoordsZoom(ARR_AS_FLOAT3(vCamForward), rho, rho_new);
}


//
//LIBRARY_API void* __stdcall GetTexture()
//{
//	return g_pTextureCurrent->GetTexture2D();
//}


LIBRARY_API void __stdcall SetTexture(void* pTex,int nTexNum)
{
	
	g_mapTextures[nTexNum] = new CTextureInfo(g_pSimpleFillTexture->GetWidth(), g_pSimpleFillTexture->GetHeight(), g_Device, (ID3D11Texture2D*)pTex,sizeof(float));
}

static void UNITY_INTERFACE_API
OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
	switch (eventType)
	{
	case kUnityGfxDeviceEventInitialize:
	{
		//TODO: user initialization code
		break;
	}
	case kUnityGfxDeviceEventShutdown:
	{
		g_RendererType = kUnityGfxRendererNull;
		//TODO: user shutdown code
		break;
	}
	case kUnityGfxDeviceEventBeforeReset:
	{
		//TODO: user Direct3D 9 code
		break;
	}
	case kUnityGfxDeviceEventAfterReset:
	{
		//TODO: user Direct3D 9 code
		break;
	}
	};
}


// Unity plugin load event
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
	g_UnityInterfaces = unityInterfaces;
	g_Graphics = unityInterfaces->Get<IUnityGraphics>();
	g_RendererType = g_Graphics->GetRenderer();
	if (g_RendererType != UnityGfxRenderer::kUnityGfxRendererD3D11)
		ReactToError("Renderer is not D3D11...");
	g_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

	auto pD3D = unityInterfaces->Get<IUnityGraphicsD3D11>();
	g_Device = pD3D->GetDevice();
	// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
	// to not miss the event in case the graphics device is already initialized
	OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

// Unity plugin unload event
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload()
{
	g_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}
