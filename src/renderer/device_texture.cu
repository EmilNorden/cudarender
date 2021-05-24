#include "device_texture.cuh"

DeviceTexture::DeviceTexture(const std::vector <uint8_t> &pixels, size_t width, size_t height)
        : m_data(nullptr), m_width(width), m_height(height) {
    cudaMalloc(&m_data, sizeof(uint8_t) * pixels.size());
    cudaMemcpy(m_data, pixels.data(), sizeof(uint8_t) * pixels.size(), cudaMemcpyHostToDevice);
}

__device__ glm::vec3 DeviceTexture::sample(const glm::vec2& uv) const {
    int x = uv.x * (m_width - 1);
    int y = uv.y * (m_height - 1);
    auto index = (y * m_width * 3) + (x*3);

    return {
            m_data[index] / 255.0f,
            m_data[index+1] / 255.0f,
            m_data[index+2] / 255.0f, // TODO: Change texture storage to floats to avoid dividing by 255 at render time?
    };
}
