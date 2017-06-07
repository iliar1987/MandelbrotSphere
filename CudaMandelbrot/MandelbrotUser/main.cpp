/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/* This example demonstrates how to use the CUDA Direct3D bindings to
* transfer data between CUDA and DX9 2D, CubeMap, and Volume Textures.
*/

#pragma warning(disable: 4312)

#include <windows.h>
#include <mmsystem.h>

#include <stdio.h>

#include <d3d11.h>
#include <d3dcompiler.h>


#include "../CudaMandelbrot1/UnityCommunicate.h"

//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
IDXGIAdapter           *g_pCudaCapableAdapter = NULL;  // Adapter to use
ID3D11Device           *g_pd3dDevice = NULL; // Our rendering device
ID3D11DeviceContext    *g_pd3dDeviceContext = NULL;
IDXGISwapChain         *g_pSwapChain = NULL; // The swap chain of the window
ID3D11RenderTargetView *g_pSwapChainRTV = NULL; //The Render target view on the swap chain ( used for clear)
ID3D11RasterizerState  *g_pRasterState = NULL;

ID3D11InputLayout      *g_pInputLayout = NULL;


//
// Vertex and Pixel shaders here : VS() & PS()
//
static const char g_simpleShaders[] =
"cbuffer cbuf \n" \
"{ \n" \
"  float4 g_vQuadRect; \n" \
"  int g_UseCase; \n" \
"} \n" \
"Texture2D g_Texture2D; \n" \
"Texture3D g_Texture3D; \n" \
"TextureCube g_TextureCube; \n" \
"\n" \
"SamplerState samLinear{ \n" \
"    Filter = MIN_MAG_LINEAR_MIP_POINT; \n" \
"};\n" \
"\n" \
"struct Fragment{ \n" \
"    float4 Pos : SV_POSITION;\n" \
"    float3 Tex : TEXCOORD0; };\n" \
"\n" \
"Fragment VS( uint vertexId : SV_VertexID )\n" \
"{\n" \
"    Fragment f;\n" \
"    f.Tex = float3( 0.f, 0.f, 0.f); \n"\
"    if (vertexId == 1) f.Tex.x = 1.f; \n"\
"    else if (vertexId == 2) f.Tex.y = 1.f; \n"\
"    else if (vertexId == 3) f.Tex.xy = float2(1.f, 1.f); \n"\
"    \n" \
"    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1);\n" \
"    \n" \
"    if (g_UseCase == 1) { \n"\
"        if (vertexId == 1) f.Tex.z = 0.5f; \n"\
"        else if (vertexId == 2) f.Tex.z = 0.5f; \n"\
"        else if (vertexId == 3) f.Tex.z = 1.f; \n"\
"    } \n" \
"    else if (g_UseCase >= 2) { \n"\
"        f.Tex.xy = f.Tex.xy * 2.f - 1.f; \n"\
"    } \n" \
"    return f;\n" \
"}\n" \
"\n" \
"float4 PS( Fragment f ) : SV_Target\n" \
"{\n" \
"    if (g_UseCase == 0) return g_Texture2D.Sample( samLinear, f.Tex.xy ); \n" \
"    else if (g_UseCase == 1) return g_Texture3D.Sample( samLinear, f.Tex ); \n" \
"    else if (g_UseCase == 2) return g_TextureCube.Sample( samLinear, float3(f.Tex.xy, 1.0) ); \n" \
"    else if (g_UseCase == 3) return g_TextureCube.Sample( samLinear, float3(f.Tex.xy, -1.0) ); \n" \
"    else if (g_UseCase == 4) return g_TextureCube.Sample( samLinear, float3(1.0, f.Tex.xy) ); \n" \
"    else if (g_UseCase == 5) return g_TextureCube.Sample( samLinear, float3(-1.0, f.Tex.xy) ); \n" \
"    else if (g_UseCase == 6) return g_TextureCube.Sample( samLinear, float3(f.Tex.x, 1.0, f.Tex.y) ); \n" \
"    else if (g_UseCase == 7) return g_TextureCube.Sample( samLinear, float3(f.Tex.x, -1.0, f.Tex.y) ); \n" \
"    else return float4(f.Tex, 1);\n" \
"}\n" \
"\n";


struct ConstantBuffer
{
	float   vQuadRect[4];
	int     UseCase;
};

ID3D11VertexShader  *g_pVertexShader;
ID3D11PixelShader   *g_pPixelShader;
ID3D11Buffer        *g_pConstantBuffer;
ID3D11SamplerState  *g_pSamplerState;

#define AssertOrQuit(x) \
    if (!(x)) \
    { \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1; \
    }

bool g_bDone = false;

const unsigned int g_WindowWidth = 720;
const unsigned int g_WindowHeight = 720;


//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd);
HRESULT InitTextures();

void RunKernels();
bool DrawScene();
void Cleanup();
void Render();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	char device_name[256];
	char *ref_file = NULL;

	printf("Starting...\n");

	//
	// create window
	//
	// Register the window class
#if 1
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
		GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
		"CUDA SDK", NULL
	};
	RegisterClassEx(&wc);

	// Create the application's window
	int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
	int yMenu = ::GetSystemMetrics(SM_CYMENU);
	int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
	HWND hWnd = CreateWindow(wc.lpszClassName, "CUDA/D3D11 Texture InterOP",
		WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu,
		NULL, NULL, wc.hInstance, NULL);
#else
	static WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, "CudaD3D9Tex", NULL };
	RegisterClassEx(&wc);
	HWND hWnd = CreateWindow(
		"CudaD3D9Tex", "CUDA D3D9 Texture Interop",
		WS_OVERLAPPEDWINDOW,
		0, 0, 800, 320,
		GetDesktopWindow(),
		NULL,
		wc.hInstance,
		NULL);
#endif

	ShowWindow(hWnd, SW_SHOWDEFAULT);
	UpdateWindow(hWnd);

	// Initialize Direct3D
	if (!SUCCEEDED(InitD3D(hWnd)) ||
		!SUCCEEDED(InitTextures()))
	{
		printf("Error initializing d3d and/or textures\r\n");
		return 1;
	}
	//
	// the main loop
	//
	while (false == g_bDone)
	{
		Render();

		//
		// handle I/O
		//
		MSG msg;
		ZeroMemory(&msg, sizeof(msg));

		while (msg.message != WM_QUIT)
		{
			if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			else
			{
				Render();
			}
		}

	};

	// Unregister windows class
	UnregisterClass(wc.lpszClassName, wc.hInstance);

	return 0;
}


//-----------------------------------------------------------------------------
// Name: InitD3D()
// Desc: Initializes Direct3D
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd)
{
	HRESULT hr = S_OK;
	cudaError cuStatus;

	// Set up the structure used to create the device and swapchain
	DXGI_SWAP_CHAIN_DESC sd;
	ZeroMemory(&sd, sizeof(sd));
	sd.BufferCount = 1;
	sd.BufferDesc.Width = g_WindowWidth;
	sd.BufferDesc.Height = g_WindowHeight;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 60;
	sd.BufferDesc.RefreshRate.Denominator = 1;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = hWnd;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;

	D3D_FEATURE_LEVEL tour_fl[] =
	{
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};
	D3D_FEATURE_LEVEL flRes;
	// Create device and swapchain
	hr = sFnPtr_D3D11CreateDeviceAndSwapChain(
		g_pCudaCapableAdapter,
		D3D_DRIVER_TYPE_UNKNOWN,//D3D_DRIVER_TYPE_HARDWARE,
		NULL, //HMODULE Software
		0, //UINT Flags
		tour_fl, // D3D_FEATURE_LEVEL* pFeatureLevels
		3, //FeatureLevels
		D3D11_SDK_VERSION, //UINT SDKVersion
		&sd, // DXGI_SWAP_CHAIN_DESC* pSwapChainDesc
		&g_pSwapChain, //IDXGISwapChain** ppSwapChain
		&g_pd3dDevice, //ID3D11Device** ppDevice
		&flRes, //D3D_FEATURE_LEVEL* pFeatureLevel
		&g_pd3dDeviceContext//ID3D11DeviceContext** ppImmediateContext
	);
	AssertOrQuit(SUCCEEDED(hr));

	g_pCudaCapableAdapter->Release();

	// Get the immediate DeviceContext
	g_pd3dDevice->GetImmediateContext(&g_pd3dDeviceContext);

	// Create a render target view of the swapchain
	ID3D11Texture2D *pBuffer;
	hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID *)&pBuffer);
	AssertOrQuit(SUCCEEDED(hr));

	hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, NULL, &g_pSwapChainRTV);
	AssertOrQuit(SUCCEEDED(hr));
	pBuffer->Release();

	g_pd3dDeviceContext->OMSetRenderTargets(1, &g_pSwapChainRTV, NULL);

	// Setup the viewport
	D3D11_VIEWPORT vp;
	vp.Width = g_WindowWidth;
	vp.Height = g_WindowHeight;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	g_pd3dDeviceContext->RSSetViewports(1, &vp);


	ID3DBlob *pShader;
	ID3DBlob *pErrorMsgs;
	// Vertex shader
	{
		D3DCompilefromfile
		D3DX11CompileFromMemory(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL, NULL,
			"VS", "vs_4_0", 0/*Flags1*/, 0/*Flags2*/, /*ID3DX11ThreadPump**/ NULL, &pShader, &pErrorMsgs, &hr);

		if (FAILED(hr))
		{
			const char *pStr = (const char *)pErrorMsgs->GetBufferPointer();
			printf(pStr);
		}

		AssertOrQuit(SUCCEEDED(hr));
		hr = g_pd3dDevice->CreateVertexShader(pShader->GetBufferPointer(), pShader->GetBufferSize(), NULL, &g_pVertexShader);
		AssertOrQuit(SUCCEEDED(hr));
		// Let's bind it now : no other vtx shader will replace it...
		g_pd3dDeviceContext->VSSetShader(g_pVertexShader, NULL, 0);
		//hr = g_pd3dDevice->CreateInputLayout(...pShader used for signature...) No need
	}
	// Pixel shader
	{
		sFnPtr_D3DX11CompileFromMemory(g_simpleShaders, strlen(g_simpleShaders), "Memory", NULL, NULL,
			"PS", "ps_4_0", 0/*Flags1*/, 0/*Flags2*/, /*ID3DX11ThreadPump**/ NULL, &pShader, &pErrorMsgs, &hr);
		AssertOrQuit(SUCCEEDED(hr));
		hr = g_pd3dDevice->CreatePixelShader(pShader->GetBufferPointer(), pShader->GetBufferSize(), NULL, &g_pPixelShader);
		AssertOrQuit(SUCCEEDED(hr));
		// Let's bind it now : no other pix shader will replace it...
		g_pd3dDeviceContext->PSSetShader(g_pPixelShader, NULL, 0);
	}
	// Create the constant buffer
	{
		D3D11_BUFFER_DESC cbDesc;
		cbDesc.Usage = D3D11_USAGE_DYNAMIC;
		cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;//D3D11_BIND_SHADER_RESOURCE;
		cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		cbDesc.MiscFlags = 0;
		cbDesc.ByteWidth = 16 * ((sizeof(ConstantBuffer) + 16) / 16);
		//cbDesc.StructureByteStride = 0;
		hr = g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pConstantBuffer);
		AssertOrQuit(SUCCEEDED(hr));
		// Assign the buffer now : nothing in the code will interfere with this (very simple sample)
		g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
		g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);
	}
	// SamplerState
	{
		D3D11_SAMPLER_DESC sDesc;
		sDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		sDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
		sDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
		sDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		sDesc.MinLOD = 0;
		sDesc.MaxLOD = 8;
		sDesc.MipLODBias = 0;
		sDesc.MaxAnisotropy = 1;
		hr = g_pd3dDevice->CreateSamplerState(&sDesc, &g_pSamplerState);
		AssertOrQuit(SUCCEEDED(hr));
		g_pd3dDeviceContext->PSSetSamplers(0, 1, &g_pSamplerState);
	}

	// Setup  no Input Layout
	g_pd3dDeviceContext->IASetInputLayout(0);
	g_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	D3D11_RASTERIZER_DESC rasterizerState;
	rasterizerState.FillMode = D3D11_FILL_SOLID;
	rasterizerState.CullMode = D3D11_CULL_FRONT;
	rasterizerState.FrontCounterClockwise = false;
	rasterizerState.DepthBias = false;
	rasterizerState.DepthBiasClamp = 0;
	rasterizerState.SlopeScaledDepthBias = 0;
	rasterizerState.DepthClipEnable = false;
	rasterizerState.ScissorEnable = false;
	rasterizerState.MultisampleEnable = false;
	rasterizerState.AntialiasedLineEnable = false;
	g_pd3dDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
	g_pd3dDeviceContext->RSSetState(g_pRasterState);

	return S_OK;
}
