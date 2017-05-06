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

	double m_R
	{
		get { return Math.Exp (- m_vecPos.magnitude * m_fRCoeff);}
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

	Vector2d GetXYRayCast()
	{
		Transform tCam = Camera.main.transform;
		RaycastHit hit;
		bool bRes = Physics.Raycast (tCam.position, tCam.forward, out hit);
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
		m_vecPos = new Vector3(0,0,1);
		m_initialRotation = gameObject.transform.rotation;
		SetNumIterations ();
		//TransformSphere ();
	}
		
	void Update () {
		
		float fForward = Input.GetAxis("Vertical");

		if (fForward != 0) {
			Transform tCam = Camera.main.transform;

			Vector2d xy_old = GetXYRayCast ();

			m_vecPos += tCam.forward * m_fSpeed * fForward;

			TransformSphere ();

			Vector2d xy_new = GetXYRayCast ();

			m_pole -= xy_new - xy_old;
			gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_xp", (float)m_pole.x);
			gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_yp", (float)m_pole.y);
			gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_R", (float)m_R);
		}

		float fTrigger = Input.GetAxis ("Trigger");
		if (fTrigger != 0) {
			m_nIterations += fTrigger;
			if (m_nIterations < 2)
				m_nIterations = 2;
			SetNumIterations ();
		}
	}
}
