#ifndef RENDERER_SCENE_H_
#define RENDERER_SCENE_H_

#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include "ray.cuh"
#include "device_mesh.cuh"
#include "device_random.cuh"
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

        std::vector<size_t> emissive_entities;
        for(auto i = 0; i < entities.size(); ++i) {
            if(entities[i].is_emissive()) {
                emissive_entities.push_back(i);
            }
        }

        cudaMalloc(&m_emissive_entity_indices, sizeof(size_t) * emissive_entities.size());
        cudaMemcpy(m_emissive_entity_indices, emissive_entities.data(), sizeof(size_t) * emissive_entities.size(), cudaMemcpyHostToDevice);
        m_emissive_entity_count = emissive_entities.size();

    }

    __device__ bool closest_intersection(const WorldSpaceRay &ray, float &out_distance) const {
        out_distance = FLT_MAX;
        auto hit_something = false;
        for (int i = 0; i < m_entity_count; ++i) {
            Intersection intersection;
            if (m_entities[i].intersect(ray, intersection)) {
                if (intersection.distance < out_distance) {
                    out_distance = intersection.distance;
                    hit_something = true;
                }
            }
        }

        return hit_something;
    }

    __device__ bool intersect(const WorldSpaceRay &ray, Intersection &intersection, SceneEntity **entity) {
        auto best_distance = FLT_MAX;
        auto success = false;

        for (int i = 0; i < m_entity_count; ++i) {
            Intersection local_intersection;
            if (m_entities[i].intersect(ray, local_intersection)) {
                if (local_intersection.distance < best_distance) {
                    best_distance = local_intersection.distance;
                    intersection = local_intersection;
                    *entity = &m_entities[i];
                    success = true;
                }
            }
        }

        return success;
    }

    __device__ SceneEntity* get_random_emissive_entity(RandomGenerator& random) {
        auto index = m_emissive_entity_indices[static_cast<size_t>(random.value() * m_emissive_entity_count)];

        return &m_entities[index];
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

                    result_color = m_entities[i].mesh()->material().sample_diffuse(
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

    size_t *m_emissive_entity_indices;
    size_t m_emissive_entity_count;
};

#endif