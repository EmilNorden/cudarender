//
// Created by emil on 2021-05-09.
//

#ifndef RENDERER_RENDERER_H
#define RENDERER_RENDERER_H

#include <GL/glew.h>

class Camera;
class Scene;

class Renderer {
public:
    Renderer(GLuint gl_texture, int width, int height);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    void render(Camera* camera, Scene* scene, int width, int height);
    void render(int width, int height, const Camera &camera, const Scene &scene);
    unsigned int* buffer() { return (unsigned int*)m_cuda_render_buffer; }
private:
    void allocate_render_buffer(int width, int height);
    void* m_cuda_render_buffer;
    struct cudaGraphicsResource* m_cuda_tex_resource;
};


#endif //RENDERER_RENDERER_H
