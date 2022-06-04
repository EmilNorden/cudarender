//
// Created by emil on 2021-05-08.
//
#ifndef RENDERER_GL_WINDOW_H
#define RENDERER_GL_WINDOW_H

#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../shader_tools/GLSLShader.h"
#include "../shader_tools/GLSLProgram.h"

struct GLFWwindow;

class GlWindow {
public:
    GlWindow(const std::string& title, int width, int height, GLFWkeyfun key_callback); // TODO: Fix the key_callback. This is a quick and dirty solution that exposes glfw to the rest of the code.

    void draw() const;
    void swap();

    [[nodiscard]] bool should_close();

    [[nodiscard]] GLFWwindow *handle() { return m_window; } // TODO: Same as above, leaking glfw too much

    void toggle_fullscreen();
private:
    GLuint VBO, VAO, EBO;
    GLFWwindow *m_window;
    GLSLShader drawtex_f; // GLSL fragment shader
    GLSLShader drawtex_v; // GLSL fragment shader
    GLSLProgram shdrawtex; // GLSLS program for textured draw
    int m_width;
    int m_height;
    bool m_is_fullscreen;
};


#endif //RENDERER_GL_WINDOW_H
