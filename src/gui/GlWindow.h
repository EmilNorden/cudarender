//
// Created by emil on 2021-05-08.
//
#ifndef RENDERER_GLWINDOW_H
#define RENDERER_GLWINDOW_H

#include <string>

struct GLFWwindow;

class GlWindow {
public:
    GlWindow(const std::string& title, int width, int height);

    void swap();

    [[nodiscard]] bool should_close();
private:
    GLFWwindow *m_window;
    int m_width;
    int m_height;
};


#endif //RENDERER_GLWINDOW_H
