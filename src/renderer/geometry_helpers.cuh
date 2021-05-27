#ifndef RENDERER_GEOMETRY_HELPERS_CUH_
#define RENDERER_GEOMETRY_HELPERS_CUH_

#include <glm/glm.hpp>

namespace geom{
    __device__ glm::vec3 get_object_space_normal_from_normal_map(
            const glm::vec3& object_space_normal,
            const glm::vec3& sampled_normal,
            const glm::vec3& object_space_tangent,
            const glm::vec3& object_space_bitangent);
}

#endif