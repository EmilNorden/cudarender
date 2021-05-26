#include "scene_entity.cuh"
#include "device_mesh.cuh"
#include "device_random.cuh"
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

bool SceneEntity::is_emissive() const {
    return m_mesh->material().emission() != glm::vec3(0, 0, 0);
}

[[nodiscard]] __device__ SurfaceDescription SceneEntity::get_random_emissive_surface(RandomGenerator& random) const
{
    auto face = m_mesh->get_random_face(random);

    float u = random.value() * 0.3f;
    float v = random.value() * 0.3f;
    float w = 1 - u - v;

    auto n0 = m_mesh->normals()[face.i0];
    auto n1 = m_mesh->normals()[face.i1];
    auto n2 = m_mesh->normals()[face.i2];

    auto world_space_normal = m_world_transform.transform_normal(
            n0 * w + n1 * u + n2 * v);

    auto v0 = m_mesh->vertices()[face.i0];
    auto v1 = m_mesh->vertices()[face.i1];
    auto v2 = m_mesh->vertices()[face.i2];

    auto world_coordinate = m_world_transform.transform_coordinate(
            v0 * w + v1 * u + v2 * v);

    return SurfaceDescription {
            m_mesh,
            world_coordinate,
            world_space_normal
    };

}