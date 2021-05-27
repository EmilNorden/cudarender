#ifndef RENDERER_DEVICE_MESH_CUH_
#define RENDERER_DEVICE_MESH_CUH_

#include <glm/glm.hpp>
#include <vector>
#include "kd_tree.cuh"
#include "ray.cuh"
#include "aabb.cuh"
#include "device_material.cuh"
#include "intersection.cuh"
#include "surface_description.cuh"

// Forward declarations
struct TreeNode;
class RandomGenerator;

struct TriangleFace {
    int i0;
    int i1;
    int i2;
};

struct Triangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};


class IndexedDeviceMesh {
public:
    __host__ IndexedDeviceMesh(const std::vector<glm::vec3> &vertices,
                               const std::vector<glm::vec3> &normals,
                               const std::vector<glm::vec3> &tangents,
                               const std::vector<glm::vec3> &bitangents,
                               const std::vector<TriangleFace> &faces,
                               const std::vector<glm::vec2> &tex_coords,
                               const DeviceMaterial &material);

    [[nodiscard]] __device__ const glm::vec3 *vertices() const { return m_vertices; }

    [[nodiscard]] __device__ int vertex_count() const { return m_vertex_count; }

    [[nodiscard]] __device__ const glm::vec3 *normals() const { return m_normals; }

    [[nodiscard]] __device__ const glm::vec3 *tangents() const { return m_tangents; }

    [[nodiscard]] __device__ const glm::vec3 *bitangents() const { return m_bitangents; }

    [[nodiscard]] __device__ const glm::vec2 *texture_coordinates() const { return m_tex_coords; }

    [[nodiscard]] __device__ int texture_coordinate_count() const { return m_tex_coord_count; }

    [[nodiscard]] __device__ __host__ DeviceMaterial &material() { return m_material; }

    [[nodiscard]] __device__ TriangleFace get_random_face(RandomGenerator &random);

    __device__ bool intersect(const ObjectSpaceRay &ray, Intersection &intersection);

private:
    DeviceMaterial m_material;
    AABB m_bounds;
    TreeNode *m_root{};

    glm::vec3 *m_vertices{};
    int m_vertex_count; // TODO is count needed on anything other than face count?
    glm::vec3 *m_normals{};
    glm::vec3 *m_tangents;
    glm::vec3 *m_bitangents;
    glm::vec2 *m_tex_coords{};
    int m_tex_coord_count;

    void build_node(TreeNode &node, std::vector<TriangleFace> &faces, Axis current_axis);
};

#endif