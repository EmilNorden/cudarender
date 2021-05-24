#ifndef RENDERER_SCENE_H_
#define RENDERER_SCENE_H_

#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include "ray.cuh"
#include "device_mesh.cuh"
#include "device_texture.cuh"

#define EPSILON 9.99999997475243E-07

class Scene {
public:
    void build(std::vector<IndexedDeviceMesh> meshes) {
        cudaMallocManaged(&m_root.meshes, sizeof(IndexedDeviceMesh) * meshes.size());
        cudaMemcpy(m_root.meshes, meshes.data(), sizeof(IndexedDeviceMesh) * meshes.size(), cudaMemcpyHostToDevice);
        m_root.mesh_count = meshes.size();
    }

    __device__ glm::vec3 hit(const WorldSpaceRay &ray) {
        auto result_color = glm::vec3(0.0f, 0.0f, 0.0f);
        auto best_distance = FLT_MAX;

        for (int i = 0; i < m_root.mesh_count; ++i) {
            float hit_distance = 0.0f;
            Intersection intersection;
            if (m_root.meshes[i].intersect(ray, intersection, hit_distance)) {
                if (hit_distance < best_distance) {
                    best_distance = hit_distance;

                    auto texcoord0 = m_root.meshes[i].texture_coordinates()[intersection.i0];
                    auto texcoord1 = m_root.meshes[i].texture_coordinates()[intersection.i1];
                    auto texcoord2 = m_root.meshes[i].texture_coordinates()[intersection.i2];

                    float w = 1.0f - intersection.u - intersection.v;

                    auto texture_uv = texcoord0 * w + texcoord1 * intersection.u + texcoord2 * intersection.v;

                    result_color = m_root.meshes[i].material().diffuse()->sample(
                            texture_uv); // glm::vec3(0.0f, 1.0f, 0.0f);
                }
            }

        }

        return result_color;
    }

private:
    struct TreeNode {
        TreeNode()
                : location(glm::vec3(0.0f, 0.0f, 0.0f)), left(nullptr), right(nullptr) {
        }

        glm::vec3 location;
        TreeNode *left;
        TreeNode *right;


        IndexedDeviceMesh *meshes;
        int mesh_count;
    };

    TreeNode m_root;
};

#endif