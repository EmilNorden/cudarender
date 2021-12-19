//
// Created by emil on 2021-05-15.
//

#ifndef RENDERER_COORDINATES_H
#define RENDERER_COORDINATES_H

#include <glm/glm.hpp>
#include "renderer_defs.h"

struct WorldSpaceCoordinate {
public:
    DEVICE_FUNC WorldSpaceCoordinate(glm::vec3 coordinate) : m_coordinate(coordinate) {}

    [[nodiscard]] DEVICE_FUNC const glm::vec3& as_vec3() const { return m_coordinate; }
private:
    glm::vec3 m_coordinate;
};

struct ObjectSpaceCoordinate {
public:
    DEVICE_FUNC ObjectSpaceCoordinate(glm::vec3 coordinate) : m_coordinate(coordinate) {}

    [[nodiscard]] DEVICE_FUNC const glm::vec3& as_vec3() const { return m_coordinate; }
private:
    glm::vec3 m_coordinate;
};

#endif //RENDERER_COORDINATES_H
