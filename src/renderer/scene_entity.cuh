#ifndef RENDERER_SCENE_ENTITY_CUH_
#define RENDERER_SCENE_ENTITY_CUH_

#include <glm/glm.hpp>
#include "ray.cuh"
#include "transform.cuh"
#include "surface_description.cuh"

class IndexedDeviceMesh;

class Intersection;

class RandomGenerator;

class SceneEntity {
public:
    SceneEntity(IndexedDeviceMesh *mesh, const WorldTransform &world_transform);

    __device__ bool intersect(const WorldSpaceRay &ray, Intersection &intersection);

    [[nodiscard]] __device__ IndexedDeviceMesh *mesh() const { return m_mesh; }

    [[nodiscard]] bool is_emissive() const;

    [[nodiscard]] __device__ const WorldTransform &world() const { return m_world_transform; }

    [[nodiscard]] __device__ const TransposeInverseWorldTransform &
    transpose_inverse_world() const { return m_transpose_inverse_world_transform; }

    [[nodiscard]] __device__ SurfaceDescription get_random_emissive_surface(RandomGenerator &random) const;

private:
    WorldTransform m_world_transform;
    InverseWorldTransform m_inverse_world_transform;
    TransposeInverseWorldTransform m_transpose_inverse_world_transform;
    IndexedDeviceMesh *m_mesh;
};

#endif