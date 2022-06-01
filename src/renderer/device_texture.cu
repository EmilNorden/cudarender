#include "device_texture.cuh"
#include <algorithm>
#include "cuda_utils.cuh"

DeviceTexture::DeviceTexture(const std::vector<uint8_t> &pixels, size_t width, size_t height)
        : m_data(nullptr), m_width(width), m_height(height) {

    std::vector<float> float_pixels;
    float_pixels.reserve(pixels.size());
    std::transform(std::begin(pixels), std::end(pixels), std::back_inserter(float_pixels), [](uint8_t value) -> float { return static_cast<float>(value) / 255.0f; });

    transfer_vector_to_device_memory(float_pixels, &m_data);
}

__device__ glm::vec3 get_color_at(float *data, size_t x, size_t y, size_t width, size_t height) {
    x = x & (width - 1);
    y = y & (height - 1);
    auto index = (y * width * 3) + (x * 3);
    return {
            data[index],
            data[index + 1],
            data[index + 2], // TODO: Change texture storage to floats to avoid dividing by 255 at render time?
    };
}

__device__ glm::vec3 DeviceTexture::sample(const glm::vec2 &uv) const {
    auto x = static_cast<size_t>(uv.x * (m_width - 1));
    auto y = static_cast<size_t>(uv.y * (m_height - 1));

    return get_color_at(m_data, x, y, m_width, m_height);
}