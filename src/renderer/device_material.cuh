#ifndef RENDERER_DEVICE_MATERIAL_CUH_
#define RENDERER_DEVICE_MATERIAL_CUH_

class DeviceTexture;

class DeviceMaterial {
public:
    __host__ DeviceMaterial(DeviceTexture *diffuse)
            : m_diffuse(diffuse) {}

    __device__ const DeviceTexture *diffuse() const { return m_diffuse; }

private:
    DeviceTexture *m_diffuse;
};

#endif