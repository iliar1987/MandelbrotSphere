#pragma once

#include "Calculator.h"

class SimpleFillTexture : public CCalculator
{
public:
	using CCalculator::CCalculator;
	virtual void UpdateBuffer(float vCamRight[3], float vCamUp[3], float vCamForward[3]) override;
};


