﻿Shader "Unlit/mandelbrot1"
{
	Properties
	{
		_R ("Projection Sphere Radius", Float) = 1.0
		_xp ("South Pole Displacement X", Float)=0.0
		_yp ("South Pole Displacement Y", Float)=0.0
		//_xp_bin ("South Pole Displacement X as pair of uints", Int)=0
		_NIterations ("Number of mandelbrot iterations", Int)=50

	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			cull back
			CGPROGRAM
			#pragma target 5.0
			#pragma vertex vert
			#pragma fragment frag
			// make fog work
			#pragma multi_compile_fog
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;
				UNITY_TRANSFER_FOG(o,o.vertex);
				return o;
			}

			static const float PI = 3.14159265358979323846264338327950288419716939937510;
			int _NIterations;
			float _xp;
			float _yp;
			float _R;

			float2 sqrComplex(float2 z)
			{
				return float2(z.x*z.x - z.y*z.y , 2*z.x * z.y);
			}

			float absSqrComplex(float2 z)
			{
				return z.x*z.x + z.y*z.y;
			}

			float3 HUEtoRGB(in float H)
			{
				float R = abs(H * 6 - 3) - 1;
				float G = 2 - abs(H * 6 - 2);
				float B = 2 - abs(H * 6 - 4);
				return saturate(float3(R,G,B));
			}

			float3 HSVtoRGB(in float3 HSV)
			{
				float3 RGB = HUEtoRGB(HSV.x);
				return ((RGB - 1) * HSV.y + 1) * HSV.z;
			}

			float3 cmap(int n)
			{
				float h = n/float(_NIterations);
				return HUEtoRGB(h);
			}

			float2 SphereProjection(float theta,float phi,float R,float xp,float yp)
			{
				return tan(theta/2.0) * 2.0 * R * float2(cos(phi),sin(phi)) + float2(xp,yp);
			}

			int Mandelbrot(float2 c)
			{
				float2 z = float2(0,0);
				for ( int i=0 ; i<_NIterations ; ++i)
				{
					z = sqrComplex(z) + c;
					if ( absSqrComplex(z) > 4 )
					{
						return i;
					}
				}
				return -1;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				float theta = (i.uv.y) * PI;
				float phi = i.uv.x * PI*2;
				float2 c = SphereProjection(theta,phi,_R,_xp,_yp);

				int nIter = Mandelbrot(c);

				fixed4 col;
				if ( nIter == -1)
					col = fixed4(0,0,0,0);
				else
					col = fixed4(cmap(nIter),1);

				UNITY_APPLY_FOG(i.fogCoord, col);
				return col;
			}
			ENDCG
		}
	}
}
