
#include "model.h"

Model::Model(std::vector<Mesh>& meshes)
    : m_meshes(std::move(meshes)){

}

Mesh::Mesh(const std::vector<glm::vec3>& vertices, const std::vector<int>& indices)
    : m_vertices(vertices), m_indices(indices) {

}