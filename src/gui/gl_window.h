//
// Created by emil on 2021-05-08.
//
#ifndef RENDERER_GL_WINDOW_H
#define RENDERER_GL_WINDOW_H

#include <string>
#include <GL/glew.h>

struct GLFWwindow;

class GlWindow {
public:
    GlWindow(const std::string& title, int width, int height);

    void draw();
    void swap();

    [[nodiscard]] bool should_close();
private:
    GLuint VBO, VAO, EBO;
    GLFWwindow *m_window;
    int m_width;
    int m_height;
};


#endif //RENDERER_GL_WINDOW_H
