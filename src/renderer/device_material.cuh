#ifndef RENDERER_DEVICE_MATERIAL_CUH_
#define RENDERER_DEVICE_MATERIAL_CUH_

class DeviceTexture;

class DeviceMaterial {
public:
    __host__ explicit DeviceMaterial(DeviceTexture *diffuse)
            : m_diffuse(diffuse) {}

    [[nodiscard]] __device__ const DeviceTexture *diffuse() const { return m_diffuse; }

    [[nodiscard]] __device__ __host__ const glm::vec3& emission() const { return m_emission; }

    __device__ __host__ void set_emission(const glm::vec3 &value) { m_emission = value; }

    [[nodiscard]] __device__ __host__ float reflectivity() const { return m_reflectivity; }

    __device__ __host__ void set_reflectivity(float value) { m_reflectivity = value; }

private:
    DeviceTexture *m_diffuse;
    glm::vec3 m_emission{};
    float m_reflectivity{};
};

#endif