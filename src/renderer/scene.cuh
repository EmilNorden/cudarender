#ifndef RENDERER_SCENE_H_
#define RENDERER_SCENE_H_

#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include "sphere.cuh"
#include "ray.cuh"
#include "device_mesh.cuh"

#define EPSILON 9.99999997475243E-07

class Scene {
public:
    __device__ glm::vec3  get_color() { return glm::vec3(1.0f, 0.0f, 0.0f); }

    void build(std::vector<Sphere> objects, std::vector<IndexedDeviceMesh> meshes) {
        cudaMallocManaged(&m_root.spheres, sizeof(Sphere) * objects.size());
        cudaMemcpy(m_root.spheres, objects.data(), sizeof(Sphere) * objects.size(), cudaMemcpyHostToDevice);
        m_root.sphere_count = objects.size();

        cudaMallocManaged(&m_root.meshes, sizeof(IndexedDeviceMesh) * meshes.size());
        cudaMemcpy(m_root.meshes, meshes.data(), sizeof(IndexedDeviceMesh) * meshes.size(), cudaMemcpyHostToDevice);
        m_root.mesh_count = meshes.size();
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

        for(int i = 0; i < m_root.mesh_count; ++i) {
            auto faces = m_root.meshes[i].faces();
            for(int j = 0; j < m_root.meshes[i].face_count(); ++j) {

                auto v0 = m_root.meshes[i].vertices()[faces[j].i0];
                auto v1 = m_root.meshes[i].vertices()[faces[j].i1];
                auto v2 = m_root.meshes[i].vertices()[faces[j].i2];

                float hit_distance = 0.0f;

                auto hit_result = hit_triangle(ray, v0, v1, v2, hit_distance);

                if(hit_result && hit_distance < best_distance) {
                    best_distance = hit_distance;
                    result_color = glm::vec3(0.0f, 1.0f, 0.0f);
                }
            }

            //float hit_distance = 0.0f;
            // m_root.meshes[i].intersect(ray, hit_distance);

        }

        return result_color;
    }

private:

    __device__ bool hit_triangle(const WorldSpaceRay& ray, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, float &out_distance) {
        // Find vectors for two edges sharing V1
        glm::vec3 e1 = v2 - v1;
        glm::vec3 e2 = v3 - v1;
        // Begin calculating determinant - also used to calculate u parameter
        glm::vec3 P = glm::cross(ray.direction(), e2); // m_direction.cross(e2);
        // if determinant is near zero, ray lies in plane of triangle

        float det = glm::dot(e1, P); // e1.dot(P);

        /*if (det > -EPSILON && det < EPSILON)
            return false;*/

        // BACK-FACE CULLING

        /*if (inside_geometry && det > EPSILON) {
            return false;
        }*/
        /*if (!inside_geometry && det < EPSILON) {
            return false;
        }*/

        float inv_det = 1.0f / det;

        // calculate distance from V1 to ray origin
        glm::vec3 T = ray.origin().as_vec3() - v1;

        // Calculate u parameter and test bound
        float u = glm::dot(T, P) * inv_det;
        // The intersection lies outside of the triangle
        if (u < 0.f || u > 1.f)
            return false;

        // Prepare to test v parameter
        glm::vec3 Q = glm::cross(T, e1); // T.cross(e1);

        // Calculate V parameter and test bound
        float v = glm::dot(ray.direction(), Q) * inv_det;
        // The intersection lies outside of the triangle
        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = glm::dot(e2, Q) * inv_det;

        if (t > EPSILON) { // ray intersection
            out_distance = t;
            // *dist = t;
            // *result_u = u;
            // *result_v = v;
            return true;
        }

        // No hit, no win
        return false;
    }

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
                :  location(glm::vec3(0.0f, 0.0f, 0.0f)), left(nullptr), right(nullptr), spheres(nullptr), sphere_count(0) {
        }

        glm::vec3 location;
        TreeNode *left;
        TreeNode *right;

        Sphere* spheres;
        int sphere_count;

        IndexedDeviceMesh *meshes;
        int mesh_count;
    };

    TreeNode m_root;
};

#endif