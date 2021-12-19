#ifndef RENDERER_RAY_H_
#define RENDERER_RAY_H_

#include <glm/glm.hpp>
#include "coordinates.cuh"
#include "transform.cuh"
#include "renderer_defs.h"

/*
template <typename CoordinateType>
class Ray {
public:
    __device__ Ray(const CoordinateType& origin, glm::vec3 direction)
        : m_origin(origin), m_direction(direction) {

    }

    [[nodiscard]] __device__ const CoordinateType& origin() const { return m_origin; }
    [[nodiscard]] __device__ const glm::vec3& direction() const { return m_direction; }
private:
    CoordinateType m_origin;
    glm::vec3 m_direction;
};

typedef Ray<WorldSpaceCoordinate> WorldSpaceRay;
typedef Ray<ObjectSpaceCoordinate> ObjectSpaceRay;*/

class Ray {
public:
    DEVICE_FUNC Ray(const glm::vec3& origin, const glm::vec3& direction)
            : m_origin(origin), m_direction(direction), m_inverse_direction({1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z}) {

    }

    [[nodiscard]] DEVICE_FUNC const glm::vec3& origin() const { return m_origin; }
    [[nodiscard]] DEVICE_FUNC const glm::vec3& direction() const { return m_direction; }
    [[nodiscard]] DEVICE_FUNC const glm::vec3& inverse_direction() const { return m_inverse_direction; }
protected:
    glm::vec3 m_origin;
    glm::vec3 m_direction;
    glm::vec3 m_inverse_direction;
};

class ObjectSpaceRay : public Ray {
public:
    DEVICE_FUNC ObjectSpaceRay(const glm::vec3& origin, const glm::vec3& direction)
            : Ray(origin, direction) {
    }
};

class WorldSpaceRay : public Ray {
public:
    DEVICE_FUNC WorldSpaceRay(const glm::vec3& origin, const glm::vec3& direction)
            : Ray(origin, direction) {
    }

    [[nodiscard]] DEVICE_FUNC ObjectSpaceRay to_object_space_ray(const InverseWorldTransform& transform) const {
        return ObjectSpaceRay{
            transform.transform_coordinate(m_origin),
            transform.transform_normal(m_direction)
        };
    }

};


#endif