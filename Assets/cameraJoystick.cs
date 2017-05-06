using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class cameraJoystick : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		float rotx = Input.GetAxis ("CamRotUp");
		float roty = Input.GetAxis ("CamRotLeft");
		if (rotx != 0 || roty != 0) {
			gameObject.transform.Rotate (new Vector3 (rotx, roty, 0));
		}

		
	}
}
