#ifndef RENDERER_DEVICE_MESH_CUH_
#define RENDERER_DEVICE_MESH_CUH_

#include <thrust/device_vector.h>
#include <glm/glm.hpp>
#include "kd_tree.cuh"

class DeviceMesh {
public:
    __host__ DeviceMesh(const std::vector<glm::vec3>& vertices, const std::vector<int>& indices);

    __device__ const int* indices() const { return m_indices; }
    __device__ int index_count() const { return m_index_count; }
    __device__ const glm::vec3* vertices() const { return m_vertices; }
    __device__ int vertex_count() const { return m_vertex_count; }

private:
    void sort_indices_by_axis(thrust::host_vector<int>& indices, Axis axis);
    glm::vec3* m_vertices;
    int m_vertex_count;
    int* m_indices;
    int m_index_count;

    // KdTree<int> indices_tree;
};

#endif