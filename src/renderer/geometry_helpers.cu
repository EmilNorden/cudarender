#include "geometry_helpers.cuh"

namespace geom {


    __device__ glm::vec3 get_object_space_normal_from_normal_map(
            const glm::vec3 &object_space_normal,
            const glm::vec3 &sampled_normal,
            const glm::vec3 &object_space_tangent,
            const glm::vec3 &object_space_bitangent) {
        glm::mat3x3 tbn{object_space_tangent.x, object_space_tangent.y,
                        object_space_tangent.z, object_space_bitangent.x,
                        object_space_bitangent.y, object_space_bitangent.z,
                        object_space_normal.x, object_space_normal.y,
                        object_space_normal.z};

        return tbn * ((sampled_normal * 2.0f) - glm::vec3{1, 1, 1});
    }

}