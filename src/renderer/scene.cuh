#ifndef RENDERER_SCENE_H_
#define RENDERER_SCENE_H_

#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include "sphere.cuh"
#include "ray.cuh"

enum Axis {
    X,
    Y,
    Z
};



class Scene {
public:
    __device__ glm::vec3  get_color() { return glm::vec3(1.0f, 0.0f, 0.0f); }

    void build(std::vector<Sphere> objects) {
        cudaMallocManaged(&m_root.spheres, sizeof(Sphere) * objects.size());
        cudaMemcpy(m_root.spheres, objects.data(), sizeof(Sphere) * objects.size(), cudaMemcpyHostToDevice);
        m_root.sphere_count = objects.size();
    }

    __device__ glm::vec3 hit(const WorldSpaceRay& ray) {
        auto result_color = glm::vec3(0.0f, 0.0f, 0.0f);
        auto best_distance = FLT_MAX;
        for(int i = 0; i < m_root.sphere_count; ++i) {
            float hit_distance = 0.0f;
            auto hit_result = hit_sphere(ray, m_root.spheres[i].location(), m_root.spheres[i].radius(), hit_distance);
            if(hit_result && hit_distance < best_distance) {
                best_distance = hit_distance;
                result_color = m_root.spheres[i].color();
            }
        }

        return result_color;
    }

private:

    __device__ bool hit_sphere(const WorldSpaceRay& ray, glm::vec3 position, float radius, float& out_distance) {

        auto squared_radius = radius*radius;
        auto L = position - ray.origin().as_vec3();
        auto tca = glm::dot(L, ray.direction());

        auto d2 = glm::dot(L, L) - tca * tca;

        if(d2 > squared_radius) {
            return false;
        }

        auto thc = glm::sqrt(squared_radius - d2);
        auto t0 = tca - thc;
        auto t1 = tca + thc;

        if(t0 > t1) {
            auto temp = t0;
            t0 = t1;
            t1 = temp;
        }

        if(t0 < 0.0) {
            t0 = t1;
            if(t0 < 0.0) {
                return false;
            }
        }

        out_distance = t0;
        return true;
    }
    struct TreeNode {
        TreeNode()
                : split_axis(Axis::X), location(glm::vec3(0.0f, 0.0f, 0.0f)), left(nullptr), right(nullptr), spheres(nullptr), sphere_count(0) {
        }

        Axis split_axis;
        glm::vec3 location;
        TreeNode *left;
        TreeNode *right;

        Sphere* spheres;
        int sphere_count;
    };

    TreeNode m_root;
};

#endif