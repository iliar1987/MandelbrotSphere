Shader "Unlit/ScreenSpaceShader"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_NIterations ("Number of Iterations",Int) = 50
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			ZWrite off
			ztest always
			Cull off
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			// make fog work

			#include "UnityCG.cginc"

			int _NIterations;

			float3 Hue2RGB(in float H)
			{
				float R = abs(H * 6 - 3) - 1;
				float G = 2 - abs(H * 6 - 2);
				float B = 2 - abs(H * 6 - 4);
				return saturate(float3(R,G,B));
			}

			float3 HSV2RGB(in float3 HSV)
			{
			    float3 RGB = Hue2RGB(HSV.x);
			    return ((RGB - 1) * HSV.y + 1) * HSV.z;
			}

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 pos : SV_POSITION;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			
			v2f vert (float2 uv : TEXCOORD0)
			{
				v2f o;
				o.uv = uv;
				o.pos = float4(o.uv*2-1,0,1);

				return o;
			}

			fixed3 cmap1(float x)
			{
				float logx = log(x+1);
				float loglogx = log(logx+1);
				return HSV2RGB(float3(
					frac(loglogx),
					1,//0.7f+0.3f*sin(2*(logx)),
					0.7f+0.3f*pow(cos(2*(logx)),3)));
			}
			fixed3 cmap2(float x)
			{
				return HSV2RGB(float3(frac(abs(x)/100),frac(abs(x)/5),1));
			}

			fixed3 cmap3(float x)
			{
				return fixed3(0.5f+0.5f*cos(x),0.5f+0.5f*sin(x),frac(x*5));
			}

			fixed3 cmap4(float x)
			{
				return 
					0.5f+0.5f 
						* sin(float3(
							1.57079632f-log(x+1)/5
							,log(x+1)
							,x/50));

			}

			fixed3 cmap5(float x)
			{
				return 0.5f+0.5f * sin(float3(x/500+3.14159265358,x/200+1.57079632f,x/20));

			}

			fixed3 cmap6(float x)
			{
				return HSV2RGB(float3(
					frac(x/211),
					1,//0.7f+0.3f*sin(2*(logx)),
					0.7f+0.3f*(sin(x/34))));

			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				float x = tex2D(_MainTex, i.uv).x;
				if ( x >= _NIterations )
					return fixed4(0,0,0,1);
				else
				{
					return fixed4(cmap6(x),1);
				}
			}
			ENDCG
		}
	}
}
