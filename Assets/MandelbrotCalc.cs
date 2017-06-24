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
	private static extern void MakeCalculation (float[] vCamRight,float[] vCamUp,float[] vCamForward, float t, float rho);

	[DllImport ("CudaMandelbrot1")]
	private static extern void Init (bool bDebug);

	[DllImport ("CudaMandelbrot1")]
	private static extern void Shutdown ();

	[DllImport ("CudaMandelbrot1")]
	private static extern void PoleCoordsGet(out float x, out float y);

	[DllImport ("CudaMandelbrot1")]
	private static extern void PoleCoordsAdd(float dx, float dy);

	[DllImport ("CudaMandelbrot1")]
	private static extern void PoleCoordsSet(float x, float y);

	int width = 1920;
	int height = 1080;
	const double PI = 3.14159265358979323846264338327950288419716939937510;

	public float m_fZoomSpeed = 1.0f;
	public float m_fRho;
	float m_fRhoInit = 1.0f;

	public float m_nIterations;
	float m_nIterationsInit = 50;
	public float m_fNIterationsGrowSpeed=1.01f;

	float t = 0;

	//private RenderTexture m_tex;
	private Texture2D m_tex;
	// Use this for initialization
	void Start () {
		m_fRho = m_fRhoInit;

		m_nIterations = m_nIterationsInit;
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
		MakeCalculation (Vec2Arr(transform.right),Vec2Arr(transform.up),Vec2Arr(transform.forward),t,m_fRho);
		FillTexture (0);
	}

	void OnApplicationQuit()
	{
		Shutdown ();
	}

	void TransformSphere()
	{
		Vector3 newForward = m_vecPos.normalized;
		Vector3 newUp = new Vector3 (0, 1, 0);
		newUp = (newUp - (Vector3.Dot (newUp, newForward)) * newForward).normalized;
		Vector3 newRight = Vector3.Cross (newUp, newForward);

		Matrix4x4 mat = new Matrix4x4();
		mat.SetColumn (0, newRight);
		mat.SetColumn (1, newUp);
		mat.SetColumn (2, newForward);
		Quaternion rotation = Quaternion.LookRotation(
			mat.GetColumn(2),
			mat.GetColumn(1)
		);

		gameObject.transform.rotation = rotation*m_initialRotation;
	}

	Vector2d SphereProjection(double theta,double phi,double R,Vector2d pole)
	{
		return new Vector2d(Math.Cos(phi),Math.Sin(phi))* (Math.Tan(theta/2.0) * 2.0 * R) + pole;
	}

	Vector2d GetXYRayCast(Vector3 Origin,Vector3 vDirection)
	{
		RaycastHit hit;
		bool bRes = Physics.Raycast (Origin, vDirection, out hit);
		Assert.IsTrue(bRes);
		Vector2 UV = hit.textureCoord;
		double theta = UV.y * PI;
		double phi = UV.x * PI * 2;
		return SphereProjection (theta, phi, m_R, m_pole);
	}

	void SetNumIterations()
	{
		gameObject.GetComponent<MeshRenderer> ().material.SetInt ("_NIterations", (int)m_nIterations);
	}

	void ModifyVecPos(Vector3 vDelta,Vector3 vOrigin,Vector3 vRayKeepConst)
	{

		Vector2d xy_old = GetXYRayCast (vOrigin,vRayKeepConst);

		m_vecPos += vDelta;

		TransformSphere ();

		Vector2d xy_new = GetXYRayCast (vOrigin,vRayKeepConst);

		m_pole -= xy_new - xy_old;

		UpdateShaderSphereProjectionParams ();
	}

	void Update () {

		float fForward = Input.GetAxis("Vertical");
		Transform tCam = Camera.main.transform;

		if (fForward != 0) {
			ModifyVecPos (tCam.forward * m_fSpeed * fForward,tCam.position,tCam.forward);
		}

		bool bRewind = Input.GetButton ("Rewind");
		if (bRewind )
		{
			if ( m_vecPos.magnitude > m_vecInitialPos.magnitude )
				ModifyVecPos (-m_vecPos.normalized * m_fSpeed,tCam.position,m_vecPos.normalized);
		}

		float fTrigger = Input.GetAxis ("Trigger");
		if (fTrigger != 0) {
			m_nIterations += fTrigger*m_fNIterationsGrowSpeed;
			if (m_nIterations < 2)
				m_nIterations = 2;
			SetNumIterations ();
		}
	}
}
