//
// Created by emil on 2021-05-09.
//

#include "coordinates.cuh"
#include "renderer.cuh"
#include "camera.cuh"

#include "cuda_runtime.h"
#include "cuda_utils.cuh"
#include <GL/glew.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_gl_interop.h>

cudaError_t cuda();

__global__ void kernel(){

}

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}

__device__ bool hit_sphere(const WorldSpaceRay& ray) {
    auto radius = 2.0f;
    auto position = glm::vec3(0.0, 0.0, 10.0);

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

    return true;
}


__global__ void
cudaRender(unsigned int *g_odata, Camera *camera, int width, int height)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

    /*Camera camera;
    camera.set_position(glm::vec3(0.0, 0.0, 0.0));
    camera.set_direction(glm::vec3(0.0, 0.0, 1.0));
    camera.set_up(glm::vec3(0.0, 1.0, 0.0));
    camera.set_field_of_view(90.0 * (3.1415 / 180.0));
    camera.set_blur_radius(0.0);
    camera.set_focal_length(1.0);
    camera.set_shutter_speed(0.0);
    camera.set_resolution(glm::vec2(width, height));

    camera.update();*/

    //uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
    // g_odata[y*width + x] = rgbToInt(c4.z, c4.y, c4.x);

    if(x < width && y < height) {
        auto ray = camera->cast_ray(x, y);

        auto hit = hit_sphere(ray);

        auto factor_x = (x / (float)width);
        auto factor_y = (y / (float)height);
        if(hit) {
            g_odata[y*width + x] = rgbToInt(factor_x * 255, 0, factor_y * 255);
        }
        else {
            g_odata[y*width + x] = rgbToInt(0, 0, 0);
        }

    }

}




void Renderer::render(Camera* camera, int width, int height) {
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, std::ceil(height / (float)block.y), 1);
    cudaRender<<<grid, block, 0>>>((unsigned int*)m_cuda_render_buffer, camera, width, height);

    cudaArray *texture_ptr;
    cuda_assert(cudaGraphicsMapResources(1, &m_cuda_tex_resource, 0));
    cuda_assert(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_cuda_tex_resource, 0, 0));

    // TODO: Havent we already calculated this?
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    cuda_assert(cudaMemcpyToArray(texture_ptr, 0, 0, m_cuda_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
    cuda_assert(cudaGraphicsUnmapResources(1, &m_cuda_tex_resource, 0));
}

Renderer::Renderer(GLuint gl_texture, int width, int height)
    : m_cuda_render_buffer(nullptr) {
    allocate_render_buffer(width, height);

    cuda_assert(cudaGraphicsGLRegisterImage(&m_cuda_tex_resource, gl_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

void Renderer::allocate_render_buffer(int width, int height) {
    if(m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }

    auto buffer_size = width * height * 4 * sizeof(GLubyte); // Is GLubyte ever larger than 1?
    cuda_assert(cudaMalloc(&m_cuda_render_buffer, buffer_size));
}

Renderer::~Renderer() {
    if(m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }
}

void Renderer::render(int width, int height, const Camera &camera, const Scene &scene)
{

}