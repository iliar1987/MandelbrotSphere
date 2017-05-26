using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;
using UnityEngine;

public class simple_test : MonoBehaviour {
	[DllImport ("CudaMandelbrot1")]
	private static extern void FillTexture();

	[DllImport ("CudaMandelbrot1")]
	private static extern IntPtr GetTexture();

	int width = 1920;
	int height = 1080;

	Texture2D m_currentTex;

	private RenderTexture m_tex;
	// Use this for initialization
	void Start () {

		//FillTexture ();


	}
	
	// Update is called once per frame
	void Update () {
		//FillTexture ();
	}

	bool bFirstFrame = true;

	void OnPreRender() {
		if (bFirstFrame) {
			bFirstFrame = false;
			IntPtr pTex=GetTexture ();
			m_currentTex = Texture2D.CreateExternalTexture (width, height, TextureFormat.RGBAFloat, false, true, pTex);
			GameObject.Find("sphere2").GetComponent<Material> ().SetTexture ("_MainTex", m_currentTex);
		}
	
	}
}
