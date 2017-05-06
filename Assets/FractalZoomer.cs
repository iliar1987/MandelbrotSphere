using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using UnityEngine.Assertions;

public class FractalZoomer : MonoBehaviour {

	Vector3 m_vecPos;
	Quaternion m_initialRotation;
	// Use this for initialization

	const double PI = 3.14159265358979323846264338327950288419716939937510;

	public float m_fSpeed = 1.0f;
	public double m_fRCoeff = 0.001;
	private Vector2d m_pole = new Vector2d(0,0);
	float m_nIterations = 50;
	public float m_fNIterationsGrowSpeed=0.3f;
	public Vector3 m_vecInitialPos = new Vector3 (0, 0, 10);
	public float m_fR0 = 2;

	double m_R
	{
		get { return Math.Exp (m_fR0 - m_vecPos.magnitude * m_fRCoeff);}
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

	void Start () {
		m_vecPos = m_vecInitialPos;
		m_initialRotation = gameObject.transform.rotation;
		UpdateShaderSphereProjectionParams ();
		SetNumIterations ();
		//TransformSphere ();
	}

	void UpdateShaderSphereProjectionParams ()
	{
		gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_xp", (float)m_pole.x);
		gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_yp", (float)m_pole.y);
		gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_R", (float)m_R);
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
