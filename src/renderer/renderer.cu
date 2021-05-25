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

    //uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
    // g_odata[y*width + x] = rgbToInt(c4.z, c4.y, c4.x);

    if (x < width && y < height) {
        // auto ray = camera->cast_ray(x, y);
        auto ray = camera->cast_perturbed_ray(x, y, random);

        //auto hit = hit_sphere(ray);
        auto color = scene->hit(ray);

        glm::vec3 previous_color;

        auto pixel_index = y * (width*4) + (x*4);

        g_odata[pixel_index] = ((g_odata[pixel_index] * (float)sample) + color.x) / (sample + 1.0f);
        g_odata[pixel_index + 1] = ((g_odata[pixel_index + 1] * (float)sample) + color.y) / (sample + 1.0f);
        g_odata[pixel_index + 2] = ((g_odata[pixel_index + 2] * (float)sample) + color.z) / (sample + 1.0f);
        g_odata[pixel_index + 3] = 1.0;

        //auto previous_r = previous_color & 0x000000
//         g_odata[y*width + x] = ((previous_color * sample) + rgbToInt(color.x, color.y, color.z)) / (sample + 1);
    }

}

void Renderer::render(Camera *camera, Scene *scene, RandomGeneratorPool *random, int width, int height, size_t sample) {
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, std::ceil(height / (float) block.y), 1);
    cudaRender<<<grid, block, 0>>>((float*) m_cuda_render_buffer, camera, scene, random, width, height, sample);

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