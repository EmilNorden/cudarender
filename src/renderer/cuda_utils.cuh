//
// Created by emil on 2021-05-09.
//

#ifndef RENDERER_CUDA_UTILS_CUH
#define RENDERER_CUDA_UTILS_CUH

#include <vector>

void cuda_assert(cudaError_t err);

template <typename T>
void transfer_vector_to_device_memory(const std::vector<T>& items, T** device_memory)
{
    cuda_assert(cudaMalloc(device_memory, sizeof(T) * items.size()));
    cuda_assert(cudaMemcpy(*device_memory, items.data(), sizeof(T) * items.size(), cudaMemcpyHostToDevice));
}

#endif //RENDERER_CUDA_UTILS_CUH
