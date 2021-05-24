#include "device_random.cuh"
#include "curand_kernel.h"

__global__ void init_random_states(curandState *states, size_t pool_size, unsigned long long seed) {
    for (auto i = 0; i < pool_size; ++i) {
        curand_init(seed, i, 0, &states[i]);
    }
}

void RandomGeneratorPool::init_global_state(size_t pool_size, unsigned long long seed) {
    init_random_states<<<1, 1>>>(m_global_state, pool_size, seed);
    cudaDeviceSynchronize();
}