#ifndef RENDERER_DEVICE_TEXTURE_CUH_
#define RENDERER_DEVICE_TEXTURE_CUH_

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

class DeviceTexture {
public:
    __host__ DeviceTexture(const std::vector<uint8_t> &pixels, size_t width, size_t height);

    [[nodiscard]] __device__ glm::vec3 sample(const glm::vec2 &uv) const;

    [[nodiscard]] __device__ glm::vec3 sample_bilinear(const glm::vec2 &uv) const;


private:
    float *m_data;
    size_t m_width;
    size_t m_height;
};

#endif