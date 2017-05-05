using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereUpdater : MonoBehaviour {
	public ComputeShader m_csMandelbrot;

	private RenderTexture m_tex;
	int width;
	int height;

	public RenderTexture GetTexture()
	{
		return m_tex;
	}



	// Use this for initialization
	void Start () {
		width = 1024;
		height = 1024;
		m_tex = new RenderTexture (width, height, 0);
		m_tex.format = RenderTextureFormat.ARGBFloat;
		m_tex.enableRandomWrite = true;
		m_tex.Create ();

		m_csMandelbrot.SetInt ("nNumIterations", 20);
		m_csMandelbrot.SetFloat ("fResolution", 0.005f);
		m_csMandelbrot.SetFloat ("fYMin", -2.0f);
		m_csMandelbrot.SetFloat ("fXMin", -2.0f);
		m_csMandelbrot.SetInt ("width", width);
		m_csMandelbrot.SetInt ("height", height);
	}
	
	// Update is called once per frame
	void Update () {
		int kernMain = m_csMandelbrot.FindKernel ("CSMain");
		m_csMandelbrot.SetTexture (kernMain, "Result", GetTexture ());
		m_csMandelbrot.Dispatch (kernMain,width / 8, height / 8, 1);

		gameObject.GetComponent<Renderer>().materials [0].SetTexture("_MainTex", GetTexture ());
	}
}
