Shader "Unlit/mandelbrot1"
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

			// Increment U
			uint4 inc128(uint4 u)
			{
			  // Compute all carries to add
			  bool4 b = (u == (uint4)(0xFFFFFFFF));
			  int4 h = (uint4)(b) * 0xFFFFFFFF;
			  uint4 c = (uint4)(h.y&h.z&h.w&1,h.z&h.w&1,h.w&1,1);
			  return u+c;
			}

			// Return -U
			uint4 neg128(uint4 u)
			{
			  return inc128(u ^ (uint4)(0xFFFFFFFF)); // (1 + ~U) is two's complement
			}

			// Return U+V
			uint4 add128(uint4 u,uint4 v)
			{
			  uint4 s = u+v;
			  uint4 h = ((uint4)(s < u)) * 0xFFFFFFFF;
			  uint4 c1 = h.yzwx & uint4(1,1,1,0); // Carry from U+V
			  h = (uint4)(s == (uint4)(0xFFFFFFFF));
			  h *= 0xFFFFFFFF;
			  uint4 c2 = (uint4)((c1.y|(c1.z&h.z))&h.y,c1.z&h.z,0,0); // Propagated carry
			  return s+c1+c2;
			}

			// Return U<<1
			uint4 shl128(uint4 u)
			{
			  uint4 h = (u>>(uint4)(31)) & uint4(0,1,1,1); // Bits to move up
			  return (u<<(uint4)(1)) | h.yzwx;
			}

			// Return U>>1
			uint4 shr128(uint4 u)
			{
			  uint4 h = (u<<(uint4)(31)) & uint4(0x80000000,0x80000000,0x80000000,0); // Bits to move down
			  return (u>>(uint4)(1)) | h.wxyz;
			}

			uint mulhilow(uint x,uint y,out uint lo,out uint hi)
			{

			}

			uint4 mul128(uint4 u,uint k)
			{

			}

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
