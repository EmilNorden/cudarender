#include "scene_entity.cuh"
#include "device_mesh.cuh"
#include "ray.cuh"
#include "transform.cuh"

SceneEntity::SceneEntity(IndexedDeviceMesh *mesh, const WorldTransform& world_transform)
    : m_world_transform(world_transform), m_inverse_world_transform(world_transform.invert()), m_mesh(mesh) {

}

__device__ bool SceneEntity::intersect(const WorldSpaceRay &ray, Intersection& intersection) {
    auto object_space_ray = ray.to_object_space_ray(m_inverse_world_transform);
    if(m_mesh->intersect(object_space_ray, intersection)) {
        auto object_space_hit_point = object_space_ray.origin() + (object_space_ray.direction() * intersection.distance);
        auto world_space_hit_point = m_world_transform.transform_coordinate(object_space_hit_point);
        intersection.distance = glm::length(world_space_hit_point - ray.origin());

        return true;
    }
    return false;
}