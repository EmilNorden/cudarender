//
// Created by emil on 2021-05-08.
//

#include "gl_window.h"
#include "glew_helper.h"
#include <GLFW/glfw3.h>
#include <string>
#include <stdexcept>

static const char *glsl_drawtex_vertshader_src =
        "#version 330 core\n"
        "layout (location = 0) in vec3 position;\n"
        "layout (location = 1) in vec3 color;\n"
        "layout (location = 2) in vec2 texCoord;\n"
        "\n"
        "out vec3 ourColor;\n"
        "out vec2 ourTexCoord;\n"
        "\n"
        "void main()\n"
        "{\n"
        "	gl_Position = vec4(position, 1.0f);\n"
        "	ourColor = color;\n"
        "	ourTexCoord = texCoord;\n"
        "}\n";

static const char *glsl_drawtex_fragshader_src =
        "#version 330 core\n"
        "uniform sampler2D tex;\n"
        "in vec3 ourColor;\n"
        "in vec2 ourTexCoord;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "   	vec4 c = texture(tex, ourTexCoord);\n"
        "   	color = c;\n"
        "}\n";


// QUAD GEOMETRY
static GLfloat vertices[] = {
        // Positions          // Colors           // Texture Coords
        1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Top Right
        1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // Bottom Right
        -1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Bottom Left
        -1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // Top Left
};
// you can also put positions, colors and coordinates in seperate VBO's
static GLuint indices[] = {  // Note that we start from 0!
        0, 1, 3,  // First Triangle
        1, 2, 3   // Second Triangle
};


GlWindow::GlWindow(const std::string& title, int width, int height, GLFWkeyfun key_callback)
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
    glfwSetKeyCallback(m_window, key_callback);


    GlewHelper::init();

    glViewport(0, 0, m_width, m_height);

    // Generate buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Buffer setup
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    // Color attribute (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    // Texture attribute (2 floats)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound
    // vertex buffer object so afterwards we can safely unbind
    glBindVertexArray(0);

    drawtex_v = GLSLShader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
    drawtex_f = GLSLShader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
    shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
    shdrawtex.compile();
}

void GlWindow::swap() {
    glfwSwapBuffers(m_window);
}

bool GlWindow::should_close() {
    return glfwWindowShouldClose(m_window);
}

void GlWindow::draw() const {
    shdrawtex.use();
    glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(VAO); // binding VAO automatically binds EBO
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0); // unbind VAO
}
