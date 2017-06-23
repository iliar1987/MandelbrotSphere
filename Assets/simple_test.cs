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
	private static extern void MakeCalculation (float[] vCamRight,float[] vCamUp,float[] vCamForward, float t, float rho);

	[DllImport ("CudaMandelbrot1")]
	private static extern void Init (bool bDebug);

	[DllImport ("CudaMandelbrot1")]
	private static extern void Shutdown ();

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

	float[] Vec2Arr(Vector3 v)
	{
		float[] arr = new float[3];
		arr [0] = v.x;
		arr [1] = v.y;
		arr [2] = v.z;
		return arr;
	}

	bool bFirstFrame = true;
	float t = 0;

	public float rho=1.0f;
	//public Quaternion q;
	void OnPreCull()
	{
		t += Time.deltaTime;
		if (bFirstFrame) {
			bFirstFrame = false;
			Init(false);
			m_tex = new Texture2D (width, height, TextureFormat.RFloat, false, false);

			IntPtr pTexPtr = m_tex.GetNativeTexturePtr ();
			SetTexture (pTexPtr,0);


			GameObject.Find ("ScreenSpaceQuad").GetComponent<MeshRenderer> ().material.SetTexture ("_MainTex",m_tex);


		}
		//q = quatRot;
		MakeCalculation (Vec2Arr(transform.right),Vec2Arr(transform.up),Vec2Arr(transform.forward),t,rho);
		FillTexture (0);
	}

	void OnApplicationQuit()
	{
		Shutdown ();
	}
}
