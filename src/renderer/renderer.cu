//
// Created by emil on 2021-05-09.
//

#include "coordinates.cuh"
#include "renderer.cuh"
#include "camera.cuh"
#include "scene.cuh"

#include "cuda_runtime.h"
#include "cuda_utils.cuh"
#include <GL/glew.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_gl_interop.h>
#include "device_random.cuh"
#include "transform.cuh"
#include "geometry_helpers.cuh"

cudaError_t cuda();

__global__ void kernel() {

}

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b) {
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b) {
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 1.0f) * 255;
    g = clamp(g, 0.0f, 1.0f) * 255;
    b = clamp(b, 0.0f, 1.0f) * 255;
    return (int(255) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
}

// convert 8-bit integer to floating point rgb color
__device__ glm::vec3 int_to_rgb(int color) {
    auto r = static_cast<float>(color & 0xFF) / 255.0f;
    auto g = static_cast<float>((color & 0xFF00) >> 8) / 255.0f;
    auto b = static_cast<float>((color & 0xFF0000) >> 16) / 255.0f;

    return glm::vec3(r, g, b);
}

__device__ bool hit_sphere(const WorldSpaceRay &ray) {
    auto radius = 2.0f;
    auto position = glm::vec3(0.0, 0.0, 10.0);

    auto squared_radius = radius * radius;
    auto L = position - ray.origin();
    auto tca = glm::dot(L, ray.direction());

    auto d2 = glm::dot(L, L) - tca * tca;

    if (d2 > squared_radius) {
        return false;
    }

    auto thc = glm::sqrt(squared_radius - d2);
    auto t0 = tca - thc;
    auto t1 = tca + thc;

    if (t0 > t1) {
        auto temp = t0;
        t0 = t1;
        t1 = temp;
    }

    if (t0 < 0.0) {
        t0 = t1;
        if (t0 < 0.0) {
            return false;
        }
    }

    return true;
}

template<typename T>
__device__ T lerp(const T &a, const T &b, float factor) {
    return a * (1.0f - factor) + b * factor;
}

__device__ glm::vec3 trace_ray(const WorldSpaceRay &ray, Scene *scene, RandomGenerator &random, int depth) {
    if (depth == 0) {
        return glm::vec3(0, 0, 0);
    }

    Intersection intersection;
    SceneEntity *entity = nullptr;
    if (scene->intersect(ray, intersection, &entity)) {

        auto intersection_coordinate = ray.origin() + (ray.direction() * intersection.distance);
        auto &material = entity->mesh()->material();
        float w = 1.0f - intersection.u - intersection.v;

        auto texcoord0 = entity->mesh()->texture_coordinates()[intersection.i0];
        auto texcoord1 = entity->mesh()->texture_coordinates()[intersection.i1];
        auto texcoord2 = entity->mesh()->texture_coordinates()[intersection.i2];

        auto texture_uv = texcoord0 * w + texcoord1 * intersection.u + texcoord2 * intersection.v;

        // Calculate normal
        auto n0 = entity->mesh()->normals()[intersection.i0];
        auto n1 = entity->mesh()->normals()[intersection.i1];
        auto n2 = entity->mesh()->normals()[intersection.i2];
        auto object_space_normal = glm::normalize(n0 * w + n1 * intersection.u + n2 * intersection.v);

        // return object_space_normal;

        if(material.has_normal_map()) {
            auto t0 = entity->mesh()->tangents()[intersection.i0];
            auto t1 = entity->mesh()->tangents()[intersection.i1];
            auto t2 = entity->mesh()->tangents()[intersection.i2];
            auto object_space_tangent = glm::normalize(t0 * w + t1 * intersection.u + t2 * intersection.v);

            auto b0 = entity->mesh()->bitangents()[intersection.i0];
            auto b1 = entity->mesh()->bitangents()[intersection.i1];
            auto b2 = entity->mesh()->bitangents()[intersection.i2];
            auto object_space_bitangent = glm::normalize(b0 * w + b1 * intersection.u + b2 * intersection.v);

            auto sampled_normal = material.sample_normal(texture_uv);

            object_space_normal = glm::normalize(geom::get_object_space_normal_from_normal_map(object_space_normal, sampled_normal, object_space_tangent, object_space_bitangent));
        }
        /*else {
            return glm::vec3(0, 1, 0);
        }*/

        auto world_space_normal = entity->world().transform_normal(object_space_normal);

        // return world_space_normal;

        // return world_space_normal;
        glm::vec3 reflected_color{};
        if (material.reflectivity() > 0.0f) {
            auto reflected_direction = glm::reflect(ray.direction(), world_space_normal);

            auto reflected_ray = WorldSpaceRay{
                    intersection_coordinate,
                    reflected_direction
            };

            reflected_color = trace_ray(reflected_ray, scene, random, depth - 1);
        }



        auto diffuse_color = entity->mesh()->material().sample_diffuse(
                texture_uv); // glm::vec3(0.0f, 1.0f, 0.0f);

        auto light = scene->get_random_emissive_entity(random);

        auto surface = light->get_random_emissive_surface(random);

        auto shadow_ray = WorldSpaceRay {
                intersection_coordinate,
                glm::normalize(surface.world_coordinate - intersection_coordinate)
        };

        glm::vec3 incoming_light{};
        Intersection shadow_intersection;
        SceneEntity *shadow_entity;
        if(scene->intersect(shadow_ray, shadow_intersection, &shadow_entity)) {
            if(shadow_entity == light) {
                incoming_light = glm::dot(world_space_normal, shadow_ray.direction()) * shadow_entity->mesh()->material().emission();
            }
        }

        return material.emission() + (incoming_light * lerp(diffuse_color, reflected_color, material.reflectivity()));
    } else {
        return glm::vec3(0, 0, 0);
    }


}


__global__ void
cudaRender(float *g_odata, Camera *camera, Scene *scene, RandomGeneratorPool *random_pool, int width, int height,
           size_t sample) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    auto threads_per_block = bw * bh;
    auto thread_num_in_block = tx + bw * ty;
    auto block_num_in_grid = blockIdx.x + gridDim.x * blockIdx.y;

    // auto global_thread_id = block_num_in_grid * threads_per_block + thread_num_in_block;
    auto global_block_id = block_num_in_grid;
    auto random = random_pool->get_generator(global_block_id);

    if (x < width && y < height) {
        auto ray = camera->cast_perturbed_ray(x, y, random);
        auto color = trace_ray(ray, scene, random, 3);


        glm::vec3 previous_color;
        auto pixel_index = y * (width * 4) + (x * 4);
        g_odata[pixel_index] = ((g_odata[pixel_index] * (float) sample) + color.x) / (sample + 1.0f);
        g_odata[pixel_index + 1] = ((g_odata[pixel_index + 1] * (float) sample) + color.y) / (sample + 1.0f);
        g_odata[pixel_index + 2] = ((g_odata[pixel_index + 2] * (float) sample) + color.z) / (sample + 1.0f);
        g_odata[pixel_index + 3] = 1.0;
    }

}

void Renderer::render(Camera *camera, Scene *scene, RandomGeneratorPool *random, int width, int height, size_t sample) {
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, std::ceil(height / (float) block.y), 1);
    cudaRender<<<grid, block, 0>>>((float *) m_cuda_render_buffer, camera, scene, random, width, height, sample);

    cudaArray *texture_ptr;
    cuda_assert(cudaGraphicsMapResources(1, &m_cuda_tex_resource, 0));
    cuda_assert(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_cuda_tex_resource, 0, 0));

    // TODO: Havent we already calculated this?
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLfloat) * num_values;
    cuda_assert(cudaMemcpyToArray(texture_ptr, 0, 0, m_cuda_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
    cuda_assert(cudaGraphicsUnmapResources(1, &m_cuda_tex_resource, 0));
}

Renderer::Renderer(GLuint gl_texture, int width, int height)
        : m_cuda_render_buffer(nullptr) {
    allocate_render_buffer(width, height);

    cuda_assert(cudaGraphicsGLRegisterImage(&m_cuda_tex_resource, gl_texture, GL_TEXTURE_2D,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
}

void Renderer::allocate_render_buffer(int width, int height) {
    if (m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }

    auto buffer_size = width * height * 4 * sizeof(GLfloat); // Is GLubyte ever larger than 1?
    cuda_assert(cudaMalloc(&m_cuda_render_buffer, buffer_size));
}

Renderer::~Renderer() {
    if (m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }
}

void Renderer::render(int width, int height, const Camera &camera, const Scene &scene) {

}