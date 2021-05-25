#ifndef RENDERER_SCENE_H_
#define RENDERER_SCENE_H_

#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include "ray.cuh"
#include "device_mesh.cuh"
#include "device_texture.cuh"
#include "scene_entity.cuh"

#define EPSILON 9.99999997475243E-07

class Scene {
public:
    void build(std::vector<IndexedDeviceMesh *> meshes, std::vector<SceneEntity> entities) {
        cudaMallocManaged(&m_meshes, sizeof(IndexedDeviceMesh *) * meshes.size());
        cudaMemcpy(m_meshes, meshes.data(), sizeof(IndexedDeviceMesh *) * meshes.size(), cudaMemcpyHostToDevice);
        m_mesh_count = meshes.size();

        cudaMallocManaged(&m_entities, sizeof(SceneEntity) * entities.size());
        cudaMemcpy(m_entities, entities.data(), sizeof(SceneEntity) * entities.size(), cudaMemcpyHostToDevice);
        m_entity_count = entities.size();
    }

    __device__ float closest_intersection(const WorldSpaceRay &ray) const {
        auto best_distance = FLT_MAX;
        for (int i = 0; i < m_entity_count; ++i) {
            Intersection intersection;
            if (m_entities[i].intersect(ray, intersection)) {
                if (intersection.distance < best_distance) {
                    best_distance = intersection.distance;
                }
            }
        }

        return best_distance;
    }

    __device__ glm::vec3 hit(const WorldSpaceRay &ray) {
        auto result_color = glm::vec3(0.0f, 0.0f, 0.0f);
        auto best_distance = FLT_MAX;

        for (int i = 0; i < m_entity_count; ++i) {
            Intersection intersection;
            if (m_entities[i].intersect(ray, intersection)) {
                if (intersection.distance < best_distance) {
                    best_distance = intersection.distance;

                    auto texcoord0 = m_entities[i].mesh()->texture_coordinates()[intersection.i0];
                    auto texcoord1 = m_entities[i].mesh()->texture_coordinates()[intersection.i1];
                    auto texcoord2 = m_entities[i].mesh()->texture_coordinates()[intersection.i2];

                    float w = 1.0f - intersection.u - intersection.v;

                    auto texture_uv = texcoord0 * w + texcoord1 * intersection.u + texcoord2 * intersection.v;

                    result_color = m_entities[i].mesh()->material().diffuse()->sample(
                            texture_uv); // glm::vec3(0.0f, 1.0f, 0.0f);
                }
            }
        }

        return result_color;
    }

private:
    IndexedDeviceMesh **m_meshes;
    int m_mesh_count;

    SceneEntity *m_entities;
    size_t m_entity_count;
};

#endif