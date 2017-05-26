#pragma once

class SimpleFillTexture
{
private:
	float4* m_d_buffer = nullptr;
	int m_width;
	int m_height;
	size_t m_pitch;

public:
	int GetWidth() const { return m_width; }
	int GetHeight() const { return m_height; }
	size_t GetPitch() const { return m_pitch; }

	SimpleFillTexture(int width,int height);
	SimpleFillTexture(const SimpleFillTexture&) = delete;
	SimpleFillTexture& operator = (const SimpleFillTexture&) = delete;

	~SimpleFillTexture();

	float4* GetCurrentBuffer() { return m_d_buffer; }
	void UpdateBuffer();
};


