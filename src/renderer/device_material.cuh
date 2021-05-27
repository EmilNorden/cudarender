#ifndef RENDERER_DEVICE_MATERIAL_CUH_
#define RENDERER_DEVICE_MATERIAL_CUH_

#include <glm/glm.hpp>

class DeviceTexture;

class DeviceMaterial {
public:
    __host__ explicit DeviceMaterial(DeviceTexture *diffuse)
            : m_diffuse(diffuse), m_uv_scale(1.0f) {}

    [[nodiscard]] __device__ bool has_normal_map() const { return m_normal_map != nullptr; }

    __device__ __host__ void set_normal_map(DeviceTexture* texture) { m_normal_map = texture; }

    [[nodiscard]] __device__ __host__ const glm::vec3& emission() const { return m_emission; }

    __device__ __host__ void set_emission(const glm::vec3 &value) { m_emission = value; }

    [[nodiscard]] __device__ __host__ float reflectivity() const { return m_reflectivity; }

    __device__ __host__ void set_reflectivity(float value) { m_reflectivity = value; }

    [[nodiscard]] __device__ __host__ float uv_scale() const { return m_uv_scale; }

    __device__ __host__ void set_uv_scale(float value) { m_uv_scale = value; }

    __device__ glm::vec3 sample_diffuse(const glm::vec2 &uv);

    __device__ glm::vec3 sample_normal(const glm::vec2 &uv);

private:
    DeviceTexture *m_diffuse;
    DeviceTexture *m_normal_map{};
    glm::vec3 m_emission{};
    float m_reflectivity{};
    float m_uv_scale;
};

#endif