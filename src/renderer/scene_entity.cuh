#ifndef RENDERER_SCENE_ENTITY_CUH_
#define RENDERER_SCENE_ENTITY_CUH_

#include <glm/glm.hpp>
#include "ray.cuh"
#include "transform.cuh"

class IndexedDeviceMesh;
class Intersection;

class SceneEntity {
public:
    SceneEntity(IndexedDeviceMesh* mesh, const WorldTransform &world_transform);

    __device__ bool intersect(const WorldSpaceRay &ray, Intersection& intersection);

    [[nodiscard]] __device__ IndexedDeviceMesh *mesh() const { return m_mesh; }
private:
    WorldTransform m_world_transform;
    InverseWorldTransform m_inverse_world_transform;
    IndexedDeviceMesh *m_mesh;
};

#endif