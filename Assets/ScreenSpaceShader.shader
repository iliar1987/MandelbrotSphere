Shader "Unlit/ScreenSpaceShader"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
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

			float3 HUEtoRGB(in float H)
			{
				float R = abs(H * 6 - 3) - 1;
				float G = 2 - abs(H * 6 - 2);
				float B = 2 - abs(H * 6 - 4);
				return saturate(float3(R,G,B));
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
			
			fixed4 frag (v2f i) : SV_Target
			{
				return fixed4(HUEtoRGB(tex2D(_MainTex, i.uv).x),1);
			}
			ENDCG
		}
	}
}
