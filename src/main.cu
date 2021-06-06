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
#include "renderer/autofocus.cuh"
#include "renderer/device_texture_loader.cuh"

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

void keyboard_func(GLFWwindow *window, int key, int scancode, int action, int mods) {}

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

void handle_input(GLFWwindow *window, Camera *camera, Scene *scene) {

    auto speed = 3.0f;
    if (glfwGetKey(window, GLFW_KEY_W)) {
        camera->set_position(camera->position() + camera->direction() * speed);
    }
    if (glfwGetKey(window, GLFW_KEY_S)) {
        camera->set_position(camera->position() - camera->direction() * speed);
    }
    if (glfwGetKey(window, GLFW_KEY_D)) {
        auto right = glm::cross(camera->up(), camera->direction());
        camera->set_position(camera->position() + right * speed);
    }
    if (glfwGetKey(window, GLFW_KEY_A)) {
        auto right = glm::cross(camera->up(), camera->direction());
        camera->set_position(camera->position() - right * speed);
    }
    if (glfwGetKey(window, GLFW_KEY_Z)) {
        camera->set_position(camera->position() + camera->up() * speed);
    }
    if (glfwGetKey(window, GLFW_KEY_X)) {
        camera->set_position(camera->position() - camera->up() * speed);
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE)) {
        device_autofocus(camera, scene, WIDTH, HEIGHT);
    }
}

double cursor_x;
double cursor_y;
bool mouselook_active = false;
bool needs_autofocus = false;

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwGetCursorPos(window, &cursor_x, &cursor_y);
            mouselook_active = true;
        } else if (action == GLFW_RELEASE) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            mouselook_active = false;
            needs_autofocus = true;
        }
    }
}

/*fn get_forward(mat: &glm::Mat4) -> glm::Vec3 {
let inverted = glm::inverse(mat);
let forward = glm::normalize(inverted[2]);
glm::vec3(forward.x, forward.y, forward.z)
}*/

glm::vec3 get_forward(const glm::mat4x4 &mat) {
    auto inverted = glm::inverse(mat);
    auto forward = glm::normalize(inverted[2]);
    return glm::vec3(forward);
}

void set_camera_direction(Camera *camera, float yaw, float pitch) {
    auto xz_rotation = glm::rotate(yaw, glm::vec3(0, 1, 0));
    auto right_vector = glm::cross(get_forward(xz_rotation), glm::vec3(0, 1, 0));
    auto final_rotation = glm::rotate(xz_rotation, pitch, right_vector);
    camera->set_direction(get_forward(final_rotation));
}

int main() {
    init_glfw();

    GlWindow window{"Hello, world!", WIDTH, HEIGHT, keyboard_func};

    init_gl_buffers();

    print_cuda_device_info();

    Renderer rend{opengl_tex_cuda, WIDTH, HEIGHT};

    auto camera = create_device_type<Camera>();

    float rot = 1.45f;
    //auto camera_position = glm::vec3(glm::cos(rot) * 10.0, 0.0000, glm::sin(rot) * 10.0f);
    auto camera_position = glm::vec3(90.0, 100.0, 200.0);
    auto camera_direction = glm::normalize(glm::vec3(0.0, 0.0, -60.0f) - camera_position);
    camera->set_position(camera_position);
    camera->set_direction(camera_direction);
    camera->set_up(glm::vec3(0.0, 1.0, 0.0));
    camera->set_field_of_view(75.0 * (3.1415 / 180.0));
    camera->set_blur_radius(0.3); // (0.03);
    camera->set_focal_length(60.0);
    camera->set_shutter_speed(0.0);
    camera->set_resolution(glm::vec2(WIDTH, HEIGHT));
    camera->update();

    DeviceMeshLoader mesh_loader;

    // auto house = mesh_loader.load("/home/emil/models/apple/apple.obj"); // 0.5 0.35 0.5
    // auto house = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    // auto suzanne = house[0];

    //std::vector<IndexedDeviceMesh> meshes;
    //meshes.push_back(suzanne);

    cudaDeviceSetLimit(cudaLimitStackSize, 2048);

    glfwPollEvents();
    auto wall = DeviceTextureLoader{}.load("/home/emil/textures/Bricks059_4K-JPG/color.jpg");
    auto wall_normal = DeviceTextureLoader{}.load("/home/emil/textures/Bricks059_4K-JPG/normal.jpg");
    auto wall_roughness = DeviceTextureLoader{}.load("/home/emil/textures/Bricks059_4K-JPG/Bricks059_4K_Roughness.jpg");
    glfwPollEvents();
    auto wood_diffuse = DeviceTextureLoader{}.load("/home/emil/textures/WoodFloor043_4K-JPG/color.jpg");
    auto wood_normal = DeviceTextureLoader{}.load("/home/emil/textures/WoodFloor043_4K-JPG/normal.jpg");

    auto nvidia_diffuse = DeviceTextureLoader{}.load("/home/emil/textures/nvidia/color.jpg");

    auto red_diffuse = DeviceTextureLoader{}.load("/home/emil/textures/Plastic007_4K-JPG/color.jpg");


    std::vector<SceneEntity> entities;

    auto floor_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    floor_mesh[0]->material().set_diffuse_map(wood_diffuse);
    floor_mesh[0]->material().set_uv_scale(glm::vec2(6.0f, 6.0f));
    floor_mesh[0]->material().set_reflectivity(0.3f);
    // floor_mesh[0]->material().set_normal_map(wood_normal);

    auto wall_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    wall_mesh[0]->material().set_diffuse_map(wall);
    wall_mesh[0]->material().set_roughness_map(wall_roughness);
    wall_mesh[0]->material().set_uv_scale(glm::vec2(4.0f, 4.0f));

    auto wall_mesh2 = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    wall_mesh2[0]->material().set_diffuse_map(wall);
    wall_mesh2[0]->material().set_uv_scale(glm::vec2(4.0f, 4.0f));
    //wall_mesh[0]->material().set_normal_map(wall_normal);

    auto crate = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    crate[0]->material().set_diffuse_map(nvidia_diffuse);
    crate[0]->material().set_uv_scale(glm::vec2(-1.0f, 1.0f));
    entities.emplace_back(
            crate[0],
            WorldTransformBuilder()
                    .with_translation({-200, 10, 200})
                    .with_uniform_scale(1.5f)
                    .build()
    );

    auto light_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    light_mesh[0]->material().set_emission(glm::vec3(1.0, 1.0, 1.0));

    auto dragon = mesh_loader.load("/home/emil/models/stanford_dragon/dragon.obj");
    dragon[0]->material().set_diffuse_map(wall);
    // dragon[0]->material().set_reflectivity(1.0f);

    // Dragon
    /*entities.emplace_back(
            dragon[0],
            WorldTransformBuilder()
            .with_translation({0.0, 0.0, -300})
                    .with_uniform_scale(20.0f)
                    .build()
    );*/

    // Ceiling Light
    /*entities.emplace_back(light_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({0.0, 960.0, 0.0})
                                  .with_scale({10.0, 0.1, 10.0})
                                  .build());*/

    entities.emplace_back(light_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({200.0, 300.0, 0.0})
                                  .with_uniform_scale(1.0)
                                  .build());


    // Mesh size is 96x96x96 cm, scaled to 960x9.6x960cm
    // Floor
    entities.emplace_back(
            floor_mesh[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 0.0, 0.0})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );

    // Ceiling
    entities.emplace_back(
            wall_mesh[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 960.0, 0.0})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );

    // Front wall
    entities.emplace_back(
            wall_mesh[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 480.0, 480.0})
                    .with_rotation({glm::pi<float>() / 2.0f, 0.0, 0.0})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );

    // Back wall
    entities.emplace_back(
            wall_mesh2[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 480.0, -480.0})
                    .with_rotation({glm::pi<float>() / 2.0f, 0.0, -glm::pi<float>() / 2.0f})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );

    // Left wall
    entities.emplace_back(
            wall_mesh[0],
            WorldTransformBuilder()
                    .with_translation({-480.0, 480.0, 0.0})
                    .with_rotation({0.0f, -glm::pi<float>() / 2.0f, glm::pi<float>() / 2.0f})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );

    // Right wall
    entities.emplace_back(
            wall_mesh[0],
            WorldTransformBuilder()
                    .with_translation({480.0, 480.0, 0.0})
                    .with_rotation({0.0, glm::pi<float>() / 2.0f, -glm::pi<float>() / 2.0f})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );


    Scene *scene;
    cudaMallocManaged(&scene, sizeof(Scene));
    new(scene) Scene;
    scene->build(entities);

    std::cout << "Creating random states..." << std::flush;
    auto random = create_device_type<RandomGeneratorPool>(2048 * 256, 123);
    std::cout << "Done." << std::endl;
    double rotation = 0.0;
    double total_duration = 0.0f;
    double max_duration = 0.0f;
    int frame_counter = 0;
    size_t sample = 0;

    glfwSetInputMode(window.handle(), GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    glfwSetMouseButtonCallback(window.handle(), mouse_button_callback);


    auto run = true;

    float yaw = 2.31;
    float pitch = 0.015f;

    set_camera_direction(camera, yaw, pitch);
    device_autofocus(camera, scene, WIDTH, HEIGHT);
    while (run && !window.should_close()) {
        handle_input(window.handle(), camera, scene);

        if (glfwGetKey(window.handle(), GLFW_KEY_ESCAPE)) {
            run = false;
        }

        if (glfwGetKey(window.handle(), GLFW_KEY_P)) {
            dragon[0]->material().set_reflectivity(0.0f);
            floor_mesh[0]->material().set_reflectivity(0.0f);
        }
        if (glfwGetKey(window.handle(), GLFW_KEY_O)) {
            dragon[0]->material().set_reflectivity(1.0f);
            floor_mesh[0]->material().set_reflectivity(0.3f);
        }

        if(glfwGetKey(window.handle(), GLFW_KEY_Y)) {
            wall_mesh2[0]->material().set_diffuse_map(red_diffuse);
            sample = 0;
        }

        if (mouselook_active) {
            set_camera_direction(camera, yaw, pitch);

            double current_cursor_x, current_cursor_y;
            glfwGetCursorPos(window.handle(), &current_cursor_x, &current_cursor_y);

            yaw = yaw - (current_cursor_x - cursor_x) * 0.01f;
            pitch = pitch + (current_cursor_y - cursor_y) * 0.01f;
            pitch = glm::clamp(pitch, -1.3f, 1.3f);
            /*float speed = 1.0f;
            camera_rotation.y += (current_cursor_x - cursor_x) / 100.0f;

            camera->set_direction(glm::vec3(glm::sin(camera_rotation.y), 0.0, glm::cos(camera_rotation.y)));*/
            cursor_x = current_cursor_x;
            cursor_y = current_cursor_y;
        }

        if(needs_autofocus) {
            needs_autofocus = false;
            device_autofocus(camera, scene, WIDTH, HEIGHT);
        }

        if (camera->needs_update()) {
            camera->update();
            sample = 0;
        }

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
                  << (total_duration / frame_counter) << "ms\t\t Max: " << max_duration << "ms\tt Sample: " << sample
                  // << "\t\tCamera: " << camera->position().x << ", " << camera->position().y << ", " << camera->position().z << ", yaw " << yaw << " pitch " << pitch
                  << "                    "
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
