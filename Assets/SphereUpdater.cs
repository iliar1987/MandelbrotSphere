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
		width = 2048;
		height = 2048;
		m_tex = new RenderTexture (width, height, 0);
		m_tex.format = RenderTextureFormat.ARGBFloat;
		m_tex.enableRandomWrite = true;
		m_tex.Create ();

		m_csMandelbrot.SetInt ("nNumIterations", 60);
		m_csMandelbrot.SetFloat ("R", 0.2f);
		m_csMandelbrot.SetFloat ("xp", -1);
		m_csMandelbrot.SetFloat ("yp", 0);
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
