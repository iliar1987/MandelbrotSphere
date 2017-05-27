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
	void OnPreCull()
	{
		if (bFirstFrame) {
			bFirstFrame = false;

			m_tex = new Texture2D (width, height, TextureFormat.RGBAFloat, false, false);

			IntPtr pTexPtr = m_tex.GetNativeTexturePtr ();
			SetTexture (pTexPtr,0);


			GameObject.Find ("sphere2").GetComponent<MeshRenderer> ().material.SetTexture ("_MainTex",m_tex);
		}
		FillTexture (0);
	}
}
