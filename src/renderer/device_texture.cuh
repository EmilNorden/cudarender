#ifndef RENDERER_DEVICE_TEXTURE_CUH_
#define RENDERER_DEVICE_TEXTURE_CUH_

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

class DeviceTexture {
public:
    __host__ DeviceTexture(const std::vector<uint8_t>& pixels, size_t width, size_t height);

    __device__ glm::vec3 sample(const glm::vec2& uv) const;


private:
    uint8_t *m_data;
    size_t m_width;
    size_t m_height;
};

#endif