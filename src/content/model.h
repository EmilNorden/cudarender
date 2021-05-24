
#ifndef RENDERER_MODEL_H
#define RENDERER_MODEL_H

#include <string>
#include <vector>
#include <glm/glm.hpp>

class Material {
public:
private:
};

class Mesh {
public:
    Mesh(const std::vector<glm::vec3>& vertices, const std::vector<int>& indices);

    const std::vector<glm::vec3>& vertices() const { return m_vertices; }
    const std::vector<int>& indices() const { return m_indices; }
private:
    std::vector<glm::vec3> m_vertices;
    std::vector<int> m_indices;
};


class Model {
public:
    explicit Model(std::vector<Mesh>& meshes);

    const std::vector<Mesh>& meshes() const { return m_meshes; }
private:
    std::vector<Mesh> m_meshes;
    std::vector<Material> m_materials;
};



#endif //RENDERER_MODEL_H
