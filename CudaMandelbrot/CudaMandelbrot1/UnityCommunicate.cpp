#include <stdexcept>

#include <Windows.h>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D11.h"

#include "main.h"

#include "TextureInfo.h"

/* passing textures to/from unity:
On Direct3D - like devices this returns a pointer to the base texture type(IDirect3DBaseTexture9 on D3D9, ID3D11Resource on D3D11, ID3D12Resource on D3D12).On OpenGL - like devices the GL texture "name" is returned; cast the pointer to integer type to get it.On Metal, the id<MTLTexture> pointer is returned.On platforms that do not support native code plugins, this function always returns NULL.
*/

IUnityInterfaces* g_UnityInterfaces = NULL;
ID3D11Device* g_Device=NULL;
IUnityGraphics* g_Graphics = NULL;
UnityGfxRenderer g_RendererType = kUnityGfxRendererNull;

CTextureInfo* g_pTextureCurrent = nullptr;

int g_width = 1920;
int g_height = 1080;

void InitTextures()
{
	g_pTextureCurrent = new CTextureInfo(g_width, g_height,g_Device);
}

void Shutdown()
{
	delete g_pTextureCurrent;
	g_pTextureCurrent = nullptr;
}

static void UNITY_INTERFACE_API
OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
	switch (eventType)
	{
	case kUnityGfxDeviceEventInitialize:
	{
		Init();
		//TODO: user initialization code
		break;
	}
	case kUnityGfxDeviceEventShutdown:
	{
		g_RendererType = kUnityGfxRendererNull;
		Shutdown();
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
		throw std::runtime_error("Renderer is not D3D11...");
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