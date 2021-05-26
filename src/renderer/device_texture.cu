#include "device_texture.cuh"

DeviceTexture::DeviceTexture(const std::vector<uint8_t> &pixels, size_t width, size_t height)
        : m_data(nullptr), m_width(width), m_height(height) {
    cudaMalloc(&m_data, sizeof(uint8_t) * pixels.size());
    cudaMemcpy(m_data, pixels.data(), sizeof(uint8_t) * pixels.size(), cudaMemcpyHostToDevice);
}

__device__ glm::vec3 get_color_at(uint8_t *data, int x, int y, size_t width) {
    auto index = (y * width * 3) + (x * 3);
    return {
            data[index] / 255.0f,
            data[index + 1] / 255.0f,
            data[index + 2] / 255.0f, // TODO: Change texture storage to floats to avoid dividing by 255 at render time?
    };
}

__device__ glm::vec3 DeviceTexture::sample(const glm::vec2 &uv) const {
    int x = uv.x * (m_width - 1);
    int y = uv.y * (m_height - 1);

    return get_color_at(m_data, x, y, m_width);
}