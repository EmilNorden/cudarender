#ifndef RENDERER_DEVICE_STACK_CUH_
#define RENDERER_DEVICE_STACK_CUH_

#include <cstddef>
#include <assert.h>

template<typename T>
class DeviceStack {
public:
    __device__ explicit DeviceStack(size_t capacity = 0)
            : m_data(nullptr), m_size(0), m_capacity(capacity) {
        cudaMalloc(&m_data, sizeof(T) * capacity);
    }

    __device__ ~DeviceStack() {
        cudaFree(m_data);
    }

    DeviceStack(const DeviceStack&) = delete;

    __device__ void push(const T &item) {
        if (m_size == m_capacity) {
            expand();
        }
        m_data[m_size++] = item;
    }

    __device__ T pop() {
        assert(m_size > 0);
        return m_data[--m_size];
    }

    [[nodiscard]] __device__ bool is_empty() const {
        return m_size == 0;
    }

private:
    T *m_data;
    size_t m_size;
    size_t m_capacity;

    __device__ void expand() {
        auto new_capacity =
                m_capacity == 0
                ? 1
                : m_capacity * 2;

        T *new_data;
        cudaMalloc(&new_data, sizeof(T) * new_capacity);
        memcpy(new_data, m_data, sizeof(T) * m_capacity);
        // cudaMemcpy(new_data, m_data, sizeof(T) * m_capacity, cudaMemcpyDeviceToDevice);
        cudaFree(m_data);
        m_data = new_data;
        m_capacity = new_capacity;
    }
};

#endif