using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;
using UnityEngine;

public class MandelbrotCalc : MonoBehaviour {
	[DllImport ("CudaMandelbrot1")]
	private static extern void FillTexture(int nTexNum);

	[DllImport ("CudaMandelbrot1")]
	private static extern void SetTexture(IntPtr pTex,int nTexNum);

	[DllImport ("CudaMandelbrot1")]
	private static extern void MakeCalculation (float[] vCamRight,float[] vCamUp,float[] vCamForward, float t, float rho,int nIterations);

	[DllImport ("CudaMandelbrot1")]
	private static extern void Init(bool bDebug, int width, int height, float FOV, string fractalName);

    [DllImport ("CudaMandelbrot1")]
	private static extern void Shutdown ();

	[DllImport ("CudaMandelbrot1")]
	private static extern void PoleCoordsGet(out float x, out float y);

//	[DllImport ("CudaMandelbrot1")]
//	private static extern void PoleCoordsAdd(float dx, float dy);

//	[DllImport ("CudaMandelbrot1")]
//	private static extern void PoleCoordsSet(float x, float y);
//
	[DllImport ("CudaMandelbrot1")]
	private static extern void PoleCoordsZoom(float[] vCamForward, float rho, float rho_new);

	public struct TPole
	{
		public float x,y;
	};

	public TPole m_Pole {
		get {
			TPole temp;
			PoleCoordsGet (out temp.x, out temp.y);
			return temp;
		}
	}

	int width = 1024;
	int height = 768;
	const double PI = 3.14159265358979323846264338327950288419716939937510;

	public float m_fZoomSpeed = 1.5f;
	public float m_fRho;
	float m_fRhoInit = 1.0f;

	public float m_nIterations;
	float m_nIterationsInit = 50;
	public float m_fNIterationsGrowSpeed=1.5f;

	float t = 0;

	//private RenderTexture m_tex;
	private Texture2D m_tex;
	// Use this for initialization
	void Start () {
		m_fRho = m_fRhoInit;
		m_nIterations = m_nIterationsInit;

//		string fractalNames[] ={"Mandelbrot","BurningShip"};
		Init(false,width,height,60*Mathf.PI/180,"burningShip");
		m_tex = new Texture2D (width, height, TextureFormat.RFloat, false, false);

		IntPtr pTexPtr = m_tex.GetNativeTexturePtr ();
		SetTexture (pTexPtr,0);


		Material mat = GameObject.Find ("ScreenSpaceQuad").GetComponent<MeshRenderer> ().material;
		mat.SetTexture ("_MainTex",m_tex);

		UpdateShaderNumIterations ();

	}

	void UpdateShaderNumIterations()
	{
		Material mat = GameObject.Find ("ScreenSpaceQuad").GetComponent<MeshRenderer> ().material;
		mat.SetInt ("_NIterations", Mathf.RoundToInt(m_nIterations));
	}

	float[] Vec2Arr(Vector3 v)
	{
		float[] arr = new float[3];
		arr [0] = v.x;
		arr [1] = v.y;
		arr [2] = v.z;
		return arr;
	}

	//bool bFirstFrame = true;

	//public Quaternion q;
	void OnPreCull()
	{
		t += Time.deltaTime;
//		if (bFirstFrame) {
//			bFirstFrame = false;
//
//
//		}
		MakeCalculation (
			Vec2Arr(transform.right),
			Vec2Arr(transform.up),
			Vec2Arr(transform.forward),
			t,
			m_fRho,
			Mathf.RoundToInt( m_nIterations));
		
		FillTexture (0);
	}

	void OnApplicationQuit()
	{
		Shutdown ();
	}

	void Update () {

		float fForward = Input.GetAxis("Vertical");
		Transform tCam = Camera.main.transform;

		if (fForward != 0) {
			float fFactor = Mathf.Pow (m_fZoomSpeed, -fForward * Time.deltaTime);
			float fOldRho = m_fRho;
			m_fRho *= fFactor;
			PoleCoordsZoom(Vec2Arr(transform.forward), fOldRho, m_fRho);
		}

//		bool bRewind = Input.GetButton ("Rewind");
//		if (bRewind )
//		{
//			if ( m_vecPos.magnitude > m_vecInitialPos.magnitude )
//				ModifyVecPos (-m_vecPos.normalized * m_fSpeed,tCam.position,m_vecPos.normalized);
//		}
		float fTrigger = Input.GetAxis ("Trigger");
		if (fTrigger != 0) {
			float fFactor = Mathf.Pow (m_fNIterationsGrowSpeed, fTrigger * Time.deltaTime);
			m_nIterations *= fFactor;
			UpdateShaderNumIterations ();
		}
	}
}
