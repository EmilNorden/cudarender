//
// Created by emil on 2021-05-09.
//

#include "renderer.cuh"

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

__global__ void
cudaRender(unsigned int *g_odata, int imgw, int offset)
{
    // extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx + offset;
    int y = blockIdx.y*bh + ty;

    uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
    g_odata[y*imgw + x] = rgbToInt(c4.z, c4.y, c4.x);
}

void launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw, int offset)
{
    //cudaRender << < grid, block, sbytes >> >(g_odata, imgw, offset);
}

void Renderer::render(int width, int height) {
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    cudaRender<<<grid, block, 0>>>((unsigned int*)m_cuda_render_buffer, width, 0);

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
