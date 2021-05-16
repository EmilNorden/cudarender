#include "camera.cuh"

#define PI 3.14159265359

__device__ WorldSpaceRay Camera::cast_ray(size_t x, size_t y) const
{
    // ray.set_origin(m_position);

    // glm::vec3 dir = (m_image_plane_origin - (m_u * m_pixel_width * (double)x) + (m_v * m_pixel_height * (double)y)) - m_position;
    // dir = glm::normalize(dir);
    // ray.set_direction(dir);
    auto direction =  (m_image_plane_origin - (m_u * m_pixel_width * (float)x) + (m_v * m_pixel_height * (float)y)) - m_position;
    direction = glm::normalize(direction);

    return WorldSpaceRay{m_position, direction};
    // ray.dist = DBL_MAX;
}

/*void Camera::cast_perturbed_ray(Ray &ray, size_t x, size_t y, std::mt19937 &rand) const
{
    cast_ray(ray, x, y);
    glm::vec3 focus_point = m_position + ray.m_direction * m_focal_length;

    std::uniform_real_distribution<float> distribution(0, 1);

    float angle = distribution(rand) * PI * 2.0f;
    float length = distribution(rand) * m_blur_radius;

    ray.m_origin = m_position + (m_u * std::sin(angle) * length) + (m_v * std::cos(angle) * length);
    ray.m_direction = glm::normalize(focus_point - ray.m_origin);
}*/

glm::vec2 Camera::project_onto_image_plane(const glm::vec3 &world_coord) const
{
    glm::vec3 coord_direction = glm::normalize(m_position - world_coord);

    float distance = glm::dot((m_image_plane_origin - m_position), m_direction) / glm::dot(coord_direction, m_direction);
    glm::vec3 image_plane_point = m_position + (distance * coord_direction);

    auto foo1 = m_image_plane_origin - image_plane_point;

    float x = glm::dot(m_u, foo1);
    float x1 = x / m_pixel_width;
    float y = glm::dot(m_v, foo1);
    float y1 = y / -m_pixel_height;

    return glm::vec2{x1, y1};
}