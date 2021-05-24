#include <iostream>
#include <cmath>
#include <chrono>

#include "shader_tools/GLSLProgram.h"
#include "shader_tools/GLSLShader.h"
#include "gui/gl_window.h"

// OpenGL
#include <GLFW/glfw3.h>

// Renderer
#include "renderer/camera.cuh"
#include "renderer/renderer.cuh"
#include "renderer/scene.cuh"
#include "content/model_loader.h"
#include "renderer/device_mesh_loader.cuh"
#include "renderer/device_random.cuh"

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

#define WIDTH   1024
#define HEIGHT  512

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
    for (int i = index; i < n; i += stride) {
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
        "uniform sampler2D tex;\n"
        "in vec3 ourColor;\n"
        "in vec2 ourTexCoord;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "   	vec4 c = texture(tex, ourTexCoord);\n"
        "   	color = c;\n"
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
    while (true) {
        const GLenum err = glGetError();
        if (err == GL_NO_ERROR) {
            break;
        }

        std::cerr << "GL Error: " << gluErrorString(err) << std::endl;
    }
}

void create_gl_texture(GLuint *gl_tex, unsigned int size_x, unsigned int size_y) {
    glGenTextures(1, gl_tex);
    glBindTexture(GL_TEXTURE_2D, *gl_tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size_x, size_y, 0, GL_RGBA, GL_FLOAT, NULL);

    check_for_gl_errors();
}

void init_glfw() {
    if (!glfwInit()) {
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

void display(Camera *camera, Scene *scene, Renderer &renderer, GlWindow &window, RandomGeneratorPool *random,
             size_t sample) {
    renderer.render(camera, scene, random, WIDTH, HEIGHT, sample);
    glfwPollEvents();


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

    shdrawtex.use();
    glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);

    window.draw();

    check_for_gl_errors();

    window.swap();
}

void print_cuda_device_info() {
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    std::cout << "Using the following CUDA device: " << std::endl;

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << (int) error_id << "\n" << cudaGetErrorString(error_id)
                  << std::endl;
        exit(1);
    }

    if (device_count == 0) {
        std::cout << "There are no available devices that support CUDA" << std::endl;
        exit(1);
    }

    int device_id = 0;

    cudaSetDevice(device_id);
    cudaDeviceProp device_properties{};
    cudaGetDeviceProperties(&device_properties, device_id);

    std::cout << "  Name: " << device_properties.name << "\n";

    int driver_version, runtime_version;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driver_version / 1000,
           (driver_version % 100) / 10, runtime_version / 1000, (runtime_version % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n\n", device_properties.major,
           device_properties.minor);

}

std::vector<TriangleFace> faces_from_indices(const std::vector<int> &indices) {
    std::vector<TriangleFace> faces;
    for (int i = 0; i < indices.size(); i += 3) {
        faces.push_back({indices[i], indices[i + 1], indices[i + 2]});
    }

    return faces;
}

template<typename T, typename... Args>
T *create_device_type(Args &&... args) {
    T *object;
    cudaMallocManaged(&object, sizeof(T));
    return new(object) T(std::forward<Args>(args)...);
}

int main() {
    init_glfw();

    GlWindow window{"Hello, world!", WIDTH, HEIGHT};

    init_gl_buffers();

    print_cuda_device_info();

    Renderer rend{opengl_tex_cuda, WIDTH, HEIGHT};

    Sphere s1{glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), 1.0f};
    Sphere s2{glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 1.0f), 1.0f};
    std::vector<Sphere> spheres;
    spheres.push_back(s1);
    spheres.push_back(s2);


    Camera *camera = create_device_type<Camera>();

    float rot = 1.45f;
    auto camera_position = glm::vec3(glm::cos(rot) * 0.0075f, 0.0000, glm::sin(rot) * 0.0075f);
    auto camera_direction = glm::normalize(glm::vec3(0.0, 0.0, 0.0f) - camera_position);
    camera->set_position(camera_position);
    camera->set_direction(camera_direction);
    camera->set_up(glm::vec3(0.0, 1.0, 0.0));
    camera->set_field_of_view(90.0 * (3.1415 / 180.0));
    camera->set_blur_radius(0.0004);
    camera->set_focal_length(0.0075);
    camera->set_shutter_speed(0.0);
    camera->set_resolution(glm::vec2(WIDTH, HEIGHT));
    camera->update();

    DeviceMeshLoader mesh_loader;
    auto meshez = mesh_loader.load("/home/emil/models/house1/black_smith.obj"); // 0.5 0.35 0.5
    // auto meshez = mesh_loader.load("/home/emil/models/apple/apple.obj"); // 0.5 0.35 0.5
    // auto meshez = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    // auto suzanne = meshez[0];

    //std::vector<IndexedDeviceMesh> meshes;
    //meshes.push_back(suzanne);
    Scene *scene;
    cudaMallocManaged(&scene, sizeof(Scene));
    new(scene) Scene;
    scene->build(spheres, meshez);

    auto random = create_device_type<RandomGeneratorPool>(2048, 123);

    double rotation = 0.0;
    double total_duration = 0.0f;
    double max_duration = 0.0f;
    int frame_counter = 0;
    size_t sample = 0;
    while (!window.should_close()) {

        /*auto camera_position = glm::vec3(glm::cos(rotation) * 0.0075f, 0.00, glm::sin(rotation) * 0.0075f);
        auto camera_direction = glm::normalize(glm::vec3(0.0, 0.0, 0.0f) - camera_position);
        camera->set_position(camera_position);
        camera->set_direction(camera_direction);
        camera->update();*/
        auto start = std::chrono::high_resolution_clock::now();
        display(camera, scene, rend, window, random, sample);
        ++sample;
        auto end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration<double, std::milli>(end - start);
        frame_counter++;
        if (frame_duration.count() > max_duration) {
            max_duration = frame_duration.count();
        }
        total_duration += frame_duration.count();
        std::cout << '\r' << "Frame time: " << frame_duration.count() << "ms\t\t Avg (10 frames): "
                  << (total_duration / frame_counter) << "ms\t\t Max: " << max_duration << "ms\tt Sample: "
                  << sample << "                    "
                  << std::flush;

        if (frame_counter == 10) {
            frame_counter = 0;
            total_duration = 0;
        }
        rotation += frame_duration.count() * 0.0005;

        check_for_gl_errors();
    }

    cudaFree(camera);

    // https://stackoverflow.com/questions/14446495/cmake-project-structure-with-unit-tests
    // https://bitbucket.org/EmilNorden/physicstracer/src/master/CMakeLists.txt
    return 0;
}
