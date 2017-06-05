#pragma once

#include "TextureFiller.h"

class SimpleFillTexture : public CTextureFiller
{
public:
	using CTextureFiller::CTextureFiller;
	virtual void UpdateBuffer(float vCamRight[3], float vCamUp[3], float vCamForward[3]) override;
};


