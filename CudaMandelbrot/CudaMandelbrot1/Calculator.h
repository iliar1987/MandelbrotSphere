#pragma once

class CCalculator
{
private:
	float* m_d_buffer = nullptr;
	const int m_width;
	const int m_height;
	const float m_FOV;
	size_t m_pitch;
protected:
	float GetFov() const { return m_FOV; }
	float* GetBuffer() { return m_d_buffer; }
public:
	int GetWidth() const { return m_width; }
	int GetHeight() const { return m_height; }
	size_t GetPitch() const { return m_pitch; }

	CCalculator(int width, int height, float FOV);
	CCalculator(const CCalculator&) = delete;
	CCalculator& operator = (const CCalculator&) = delete;

	virtual ~CCalculator();

	virtual float* GetCurrentBuffer() { return m_d_buffer; }
	virtual void UpdateBuffer(float vCamRight[3], float vCamUp[3], float vCamForward[3]) = 0;
};
