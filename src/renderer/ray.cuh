#ifndef RENDERER_RAY_H_
#define RENDERER_RAY_H_

#include <glm/glm.hpp>
#include "coordinates.cuh"

template <typename CoordinateType>
class Ray {
public:
    __device__ Ray(const CoordinateType& origin, glm::vec3 direction)
        : m_origin(origin), m_direction(direction) {

    }

    __device__ const CoordinateType& origin() const { return m_origin; }
    __device__ const glm::vec3& direction() const { return m_direction; }
private:
    CoordinateType m_origin;
    glm::vec3 m_direction;
};

typedef Ray<WorldSpaceCoordinate> WorldSpaceRay;

#endif