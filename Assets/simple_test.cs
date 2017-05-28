using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;
using UnityEngine;

public class simple_test : MonoBehaviour {
	[DllImport ("CudaMandelbrot1")]
	private static extern void FillTexture(int nTexNum);

	[DllImport ("CudaMandelbrot1")]
	private static extern void SetTexture(IntPtr pTex,int nTexNum);

	[DllImport ("CudaMandelbrot1")]
	private static extern void MakeCalculation (float v0, float v1, float v2, float v3);

	int width = 1920;
	int height = 1080;

	//private RenderTexture m_tex;
	private Texture2D m_tex;
	// Use this for initialization
	void Start () {
//		m_tex = new RenderTexture (width, height, 0, RenderTextureFormat.ARGBFloat);
//		m_tex.enableRandomWrite = true;
//		m_tex.Create ();



		//gameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex",m_tex);
	}

	// Update is called once per frame
	void Update () {
		//FillTexture ();
	}

	bool bFirstFrame = true;
	//public Quaternion q;
	void OnPreCull()
	{
		if (bFirstFrame) {
			bFirstFrame = false;

			m_tex = new Texture2D (width, height, TextureFormat.RGBAFloat, false, false);

			IntPtr pTexPtr = m_tex.GetNativeTexturePtr ();
			SetTexture (pTexPtr,0);


			GameObject.Find ("ScreenSpaceQuad").GetComponent<MeshRenderer> ().material.SetTexture ("_MainTex",m_tex);
		}
		Quaternion quatRot = transform.rotation;
		//q = quatRot;
		MakeCalculation (-quatRot.x,-quatRot.y,-quatRot.z,quatRot.w);
		FillTexture (0);
	}
}
