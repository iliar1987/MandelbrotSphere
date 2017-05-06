using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FractalZoomer : MonoBehaviour {

	Vector3 m_vecPos;
	// Use this for initialization
	void Start () {
		m_vecPos = new Vector3(0,0,0);

	}
	const float PI = 3.14159265f;

	public float m_fSpeed = 0.001f;
	public float m_fRCoeff = 0.001f;
	private Vector2 m_fPole = new Vector2(0,0);
	// Update is called once per frame

	float CalcR(Vector3 vecPos)
	{
		return Mathf.Exp (- vecPos.magnitude * m_fRCoeff);
	}

	void Update () {
		
		float fForward = Input.GetAxis("Vertical");

		if (fForward > 0) {
			Transform tCam = Camera.main.transform;

			RaycastHit hit;
			if (Physics.Raycast (tCam.position, tCam.forward, out hit)) {
				Vector2 UV = hit.textureCoord;
				float theta = UV.y * PI;
				float phi = UV.x * PI * 2;
				float R_old = CalcR (m_vecPos);

				m_vecPos += tCam.forward * m_fSpeed;
				float R_new = CalcR (m_vecPos);
				m_fPole += 2 * Mathf.Tan ((PI - theta) / 2) 
					* (new Vector2 (Mathf.Cos (phi), Mathf.Sin (phi)))
					* (R_old-R_new);
				gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_xp", m_fPole.x);
				gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_yp", m_fPole.y);
				gameObject.GetComponent<MeshRenderer> ().material.SetFloat ("_R", R_new);
			}
		}
	}
}
