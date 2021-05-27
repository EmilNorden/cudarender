#include "device_material.cuh"
#include "device_texture.cuh"

__device__ glm::vec3 DeviceMaterial::sample_diffuse(const glm::vec2 &uv)
{
    return m_diffuse->sample(uv * m_uv_scale);
}

__device__ glm::vec3 DeviceMaterial::sample_normal(const glm::vec2 &uv)
{
    return m_normal_map->sample(uv * m_uv_scale);
}