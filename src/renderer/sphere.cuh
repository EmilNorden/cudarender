
#ifndef RENDERER_SPHERE_H
#define RENDERER_SPHERE_H

#include <glm/glm.hpp>

class Sphere {
public:
    __device__ __host__  Sphere(const glm::vec3& location, const glm::vec3& color, float radius)
        : m_location(location), m_color(color), m_radius(radius) {
    }

    __device__ __host__ const glm::vec3& location() const { return m_location; }
    __device__ __host__ const glm::vec3& color() const { return m_color; }
    __device__ __host__ float radius() const { return m_radius; }
private:
    glm::vec3 m_location;
    glm::vec3 m_color;
    float m_radius;
};

#endif //RENDERER_SPHERE_H
