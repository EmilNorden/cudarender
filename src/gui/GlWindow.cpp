//
// Created by emil on 2021-05-08.
//

#include "GlWindow.h"
#include <GLFW/glfw3.h>
#include <string>
#include <string_view>
#include <stdexcept>


void keyboard_func(GLFWwindow* window, int key, int scancode, int action, int mods){
}


GlWindow::GlWindow(const std::string& title, int width, int height)
    : m_window(nullptr), m_width(width), m_height(height) {

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(m_width, m_height, title.c_str(), nullptr, nullptr);
    if(!m_window) {
        throw std::runtime_error("Unable to create window!");
    }

    // Not sure the constructor should be doing this
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(m_window, keyboard_func);
}

void GlWindow::swap() {
    glfwSwapBuffers(m_window);
}

bool GlWindow::should_close() {
    return glfwWindowShouldClose(m_window);
}
