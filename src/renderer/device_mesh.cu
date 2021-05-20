#include "device_mesh.cuh"

__host__ DeviceMesh::DeviceMesh(const std::vector<glm::vec3>& vertices, const std::vector<int>& indices){

    cudaMalloc(&m_vertices, sizeof(glm::vec3) * vertices.size());
    cudaMemcpy(m_vertices, vertices.data(), sizeof(glm::vec3)*vertices.size(), cudaMemcpyHostToDevice);
    m_vertex_count = vertices.size();

    cudaMalloc(&m_indices, sizeof(int) * indices.size());
    cudaMemcpy(m_indices, indices.data(), sizeof(int)*indices.size(), cudaMemcpyHostToDevice);
    m_index_count = indices.size();
}

void DeviceMesh::sort_indices_by_axis(thrust::host_vector<int>& indices, Axis axis) {
    std::sort(indices.begin(), indices.end(), [] (int& index1, int& index2) {

        return index1 > index2;
    });
}