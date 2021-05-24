#ifndef RENDERER_DEVICE_RANDOM_CUH_
#define RENDERER_DEVICE_RANDOM_CUH_

#include <curand.h>

class DeviceRandom {
public:
    __host__ explicit DeviceRandom(size_t count) {

    }
private:
    size_t m_count;
};

#endif