#ifndef RENDERER_DEVICE_MESH_CUH_
#define RENDERER_DEVICE_MESH_CUH_

#include <thrust/device_vector.h>
#include <glm/glm.hpp>
#include <vector>
#include "kd_tree.cuh"
#include "ray.cuh"
#include "aabb.cuh"

// Forward declarations
struct TreeNode;

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
    __host__ IndexedDeviceMesh(const std::vector<glm::vec3>& vertices, const std::vector<TriangleFace>& faces);

    __device__ const TriangleFace* faces() const { return m_faces; }
    __device__ int face_count() const { return m_face_count; }
    __device__ const glm::vec3* vertices() const { return m_vertices; }
    __device__ int vertex_count() const { return m_vertex_count; }

    __device__ bool intersect(const WorldSpaceRay& ray, float &out_distance);

private:

    AABB m_bounds;
    TreeNode *m_root;

    std::vector<glm::vec3> m_host_vertices;
    glm::vec3* m_vertices;
    int m_vertex_count;
    TriangleFace* m_faces;
    int m_face_count;

    void build_node(TreeNode& node, std::vector<TriangleFace>& faces, Axis current_axis);
};

#endif