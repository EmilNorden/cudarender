//
// Created by emil on 2021-05-09.
//

#ifndef RENDERER_RENDERER_H
#define RENDERER_RENDERER_H


class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    void render(unsigned int *out_buffer, int width, int height);
private:
    void allocate_render_buffer(int width, int height);
    void* m_cuda_render_buffer;
};


#endif //RENDERER_RENDERER_H
