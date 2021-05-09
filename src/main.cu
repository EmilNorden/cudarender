#include <iostream>
#include <cmath>
#include <chrono>

#include "shader_tools/GLSLProgram.h"
#include "shader_tools/GLSLShader.h"
#include "gui/gl_window.h"
#include "renderer/renderer.cuh"

// OpenGL
#include <GLFW/glfw3.h>

#if defined(RENDER_DEBUG)
#define DEBUG_ASSERT_SDL(x) {                                   \
                                if((x) == -1) {                 \
                                    std::cerr                   \
                                        << "SDL call failed: "  \
                                        << SDL_GetError()       \
                                    exit(1);                    \
                                }                               \
                            }
#else
#define DEBUG_ASSERT_SDL(x) (x)
#endif

// OpenGL
// GLuint VBO, VAO, EBO;
GLSLShader drawtex_f; // GLSL fragment shader
GLSLShader drawtex_v; // GLSL fragment shader
GLSLProgram shdrawtex; // GLSLS program for textured draw

// CUDA <-> OpenGL interop
GLuint opengl_tex_cuda;

#define WIDTH   800
#define HEIGHT  600

#if defined(RENDER_DEBUG)
#define DEBUG_ASSERT_SDL_PTR(x) {                                   \
                                    if(!(x)) {                      \
                                        std::cerr                   \
                                            << "SDL call failed: "  \
                                            << SDL_GetError()       \
                                        exit(1);                    \
                                    }                               \
                                }
#else
#define DEBUG_ASSERT_SDL_PTR(x)
#endif
__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

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
        "uniform usampler2D tex;\n"
        "in vec3 ourColor;\n"
        "in vec2 ourTexCoord;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "   	vec4 c = texture(tex, ourTexCoord);\n"
        "   	color = c / 255.0;\n"
        "}\n";

/*
// QUAD GEOMETRY
GLfloat vertices[] = {
        // Positions          // Colors           // Texture Coords
        1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Top Right
        1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // Bottom Right
        -1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Bottom Left
        -1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // Top Left
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {  // Note that we start from 0!
        0, 1, 3,  // First Triangle
        1, 2, 3   // Second Triangle
};
*/

void check_for_gl_errors() {
    while(true) {
        const GLenum err = glGetError();
        if(err == GL_NO_ERROR) {
            break;
        }

        std::cerr << "GL Error: " << gluErrorString(err) << std::endl;
    }
}

void create_gl_texture(GLuint* gl_tex, unsigned int size_x, unsigned int size_y) {
    glGenTextures(1, gl_tex);
    glBindTexture(GL_TEXTURE_2D, *gl_tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);

    check_for_gl_errors();
}

void init_glfw() {
    if(!glfwInit()) {
        std::cerr << "glfwInit failed!" << std::endl;
        exit(1);
    }
}

void init_gl_buffers() {
    create_gl_texture(&opengl_tex_cuda, WIDTH, HEIGHT);

    drawtex_v = GLSLShader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
    drawtex_f = GLSLShader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
    shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
    shdrawtex.compile();
    check_for_gl_errors();
}

void display(Renderer& renderer, GlWindow& window, int frame) {
    renderer.render(WIDTH, HEIGHT);
    glfwPollEvents();


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

    shdrawtex.use();
    glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);

    window.draw();

    check_for_gl_errors();

    window.swap();
}


int main() {
    init_glfw();

    GlWindow window{"Hello, world!", WIDTH, HEIGHT};

    init_gl_buffers();

    Renderer rend{opengl_tex_cuda, WIDTH, HEIGHT};

    int frame = 0;
    while(!window.should_close()) {
        auto start = std::chrono::high_resolution_clock::now();
        display(rend, window, frame);
        //glfwWaitEvents();
        frame++;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end-start);
        std::cout << "Time to render 1 frame: " << duration.count() << std::endl;
        // std::cout << "FPS: " << (1000.0 / duration.count()) << std::endl;
    }

    int N = 1<<20;
    float *x, *y;

    // Allocate unified memory - accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for(int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for(int i = 0; i < N; i++) {
        maxError = std::max(maxError, std::abs(y[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    // https://stackoverflow.com/questions/14446495/cmake-project-structure-with-unit-tests
    // https://bitbucket.org/EmilNorden/physicstracer/src/master/CMakeLists.txt
    return 0;
}
