#pragma once

#include "TextureFiller.h"

class SimpleFillTexture : public CTextureFiller
{
private:
	virtual void LaunchKernel(const KernelParameters& params) override;
public:
	using CTextureFiller::CTextureFiller;
	
};


