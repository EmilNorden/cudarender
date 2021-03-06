#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

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
#include "renderer/device_material_loader.cuh"
#include "renderer/cuda_utils.cuh"

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


// CUDA <-> OpenGL interop
GLuint opengl_tex_cuda;

#define WIDTH   1024
#define HEIGHT  512


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
    check_for_gl_errors();
}

void display(Camera *camera, Scene *scene, Renderer &renderer, GlWindow &window, RandomGeneratorPool *random,
             size_t sample) {
    renderer.render(camera, scene, random, WIDTH, HEIGHT, sample);
    glfwPollEvents();


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);


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

    printf("  CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driver_version / 1000,
           (driver_version % 100) / 10, runtime_version / 1000, (runtime_version % 100) / 10);
    printf("  CUDA Capability Major/Minor version number: %d.%d\n", device_properties.major,
           device_properties.minor);
    printf("  SM Count: %d, Warp size: %d, Shared mem/block %zu \n\n", device_properties.multiProcessorCount, device_properties.warpSize, device_properties.sharedMemPerBlock);

}

void handle_input(GlWindow &gl_window, Camera *camera, Scene *scene) {
    auto window = gl_window.handle();
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
    if(glfwGetKey(window, GLFW_KEY_F12)) {
        gl_window.toggle_fullscreen();
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

void
scene_dragon(DeviceMeshLoader &mesh_loader, DeviceMaterialLoader &material_loader, DeviceTextureLoader &texture_loader,
             std::vector<SceneEntity> &entities, std::mt19937 &rand) {
    /*auto interior = mesh_loader.load("/home/emil/models/Stockholm_interior_OBJ/Stockholm_interior.obj");

    entities.emplace_back(interior[0],
                          WorldTransformBuilder()
                                  .with_translation({0.0, 0.0, 0.0})
                                  .with_scale({1.0, 0.1f, 1.0f})
                                  .build());*/
    auto paper_material = material_loader.load("/home/emil/textures/gray/");

    auto dragon = mesh_loader.load("/home/emil/models/stanford_dragon/dragon.obj");
    paper_material.set_roughness_map(nullptr);
    // paper_material.set_reflectivity(1.0f);
    paper_material.set_normal_map(nullptr);
    //paper_material.set_emission(glm::vec3(1,1,1) * 10000.0f);
    dragon[0]->set_material(paper_material);

    //auto box = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    //auto nvidia = material_loader.load("/home/emil/textures/nvidia/");
    //nvidia.set_uv_scale({-1, 1});
    //box[0]->set_material(nvidia);
    auto light_mesh = mesh_loader.load("/home/emil/models/woodsphere/wooden_sphere.obj");

    light_mesh[0]->material().set_emission(glm::vec3(1.0, 1.0, 1.0) * 120000000.0f);
    entities.emplace_back(light_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({10000.0, 10000.0, 0.0})
                                  .with_uniform_scale(100)
                                  .build());

    auto ball_mesh = mesh_loader.load("/home/emil/models/woodsphere/wooden_sphere.obj");
    ball_mesh[0]->material().set_reflectivity(1.0f);
    entities.emplace_back(ball_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({200.0, 12.0, 0.0})
                                  .with_uniform_scale(10)
                                  .build());

    entities.emplace_back(ball_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({100.0, 12.0, -200.0})
                                  .with_uniform_scale(10)
                                  .build());

    entities.emplace_back(ball_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({0.0, 12.0, 150.0})
                                  .with_uniform_scale(10)
                                  .build());


    auto floor_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    auto metal_mat = material_loader.load("/home/emil/textures/Metal004_4K-JPG/");
    metal_mat.set_uv_scale({100.0, 100.0});
    // grass.set_normal_map(nullptr);
    //grass.set_roughness_map(roughness);
    metal_mat.set_reflectivity(0.75f);

    floor_mesh[0]->set_material(metal_mat);
    entities.emplace_back(
            floor_mesh[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 0.0, 0.0})
                    .with_scale({100.0, 0.1, 100.0})
                    .build()
    );
    entities.emplace_back(
            dragon[0],
            WorldTransformBuilder()
                    .with_translation({0, 0.0, 0})
                    .with_uniform_scale(15.0f)
                    .build()
    );

    /*std::uniform_real_distribution<> dis(0.0, 1.0);
    auto hspacing = 0.0f;
    auto yspacing = 0.0f;
    int count =3;
    for (int x = 0; x < count; ++x) {

        yspacing += 50.0f + dis(rand) * 60.0f;
        hspacing = 0;
        for (int y = 0; y < count; ++y) {

            //paper_material.set_reflectivity(dis(rand));
            //dragon[0]->set_material(paper_material);
            hspacing += 50.0f + dis(rand) * 60.0f;
            entities.emplace_back(
                    dragon[0],
                    WorldTransformBuilder()
                            .with_translation({x * 105 + yspacing, 0.0, y * 105 + hspacing})
                            .with_uniform_scale(15.0f)
                            .build()
            );
        }
    }*/
}


void
scene_house(DeviceMeshLoader &mesh_loader, DeviceMaterialLoader &material_loader, DeviceTextureLoader &texture_loader,
            std::vector<SceneEntity> &entities) {
    auto light_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    light_mesh[0]->material().set_emission(glm::vec3(1.0, 1.0, 1.0) * 90000.0f);
    entities.emplace_back(light_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({0.0, 500.0, 0.0})
                                  .with_scale({1.0f, 1.0f, 1.0f})
                                  .build());

    auto house = mesh_loader.load("/home/emil/models/house1/black_smith.obj");

    entities.emplace_back(
            house[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 248.1, 0.0})
                    .with_uniform_scale(300.0f)
                    .build()
    );

    auto floor_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    auto grass = material_loader.load("/home/emil/textures/Grass001_4K-JPG/");
    grass.set_uv_scale({100.0, 100.0});
    floor_mesh[0]->set_material(grass);
    entities.emplace_back(
            floor_mesh[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 0.0, 0.0})
                    .with_scale({100.0, 0.1, 100.0})
                    .build()
    );
}

void scene_wall_lamps(DeviceMeshLoader &mesh_loader, DeviceMaterialLoader &material_loader,
                      DeviceTextureLoader &texture_loader, std::vector<SceneEntity> &entities) {
    auto wall_material = material_loader.load("/home/emil/textures/Bricks059_4K-JPG/");
    wall_material.set_normal_map(nullptr);
    wall_material.set_roughness_map(nullptr);
    wall_material.set_uv_scale(glm::vec2(4.0f, 4.0f));
    // auto wood_material = material_loader.load("/home/emil/textures/WoodFloor043_4K-JPG/");
    auto paper_material = material_loader.load("/home/emil/textures/Paper001_4K-JPG/");

    auto nvidia_texture = texture_loader.load("/home/emil/textures/nvidia/color.png");
    auto black_mat = material_loader.load("/home/emil/textures/black/");


    auto wall_lamp = mesh_loader.load("/home/emil/models/wall_lamp/lamp.obj");
    wall_lamp[0]->set_material(wall_material);
    wall_lamp[1]->set_material(wall_material);
    wall_lamp[2]->set_material(wall_material);
    wall_lamp[2]->material().set_emission({1.0, 1.0, 1.0});
    wall_lamp[3]->set_material(wall_material);


    auto floor_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    floor_mesh[0]->set_material(wall_material);

    auto wall_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    wall_mesh[0]->set_material(wall_material);

    auto wall_mesh2 = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    wall_mesh2[0]->set_material(wall_material);

    auto crate = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    crate[0]->material().set_diffuse_map(nvidia_texture);
    crate[0]->material().set_uv_scale(glm::vec2(-1.0f, 1.0f));
    entities.emplace_back(
            crate[0],
            WorldTransformBuilder()
                    .with_translation({-200, 10, 200})
                    .with_uniform_scale(1.5f)
                    .build()
    );

    auto light_mesh = mesh_loader.load("/home/emil/models/crate/crate1.obj");
    light_mesh[0]->material().set_emission(glm::vec3(1.0, 1.0, 1.0) * 1000000.0f);
    entities.emplace_back(light_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({200.0, 300.0, 0.0})
                                  .with_uniform_scale(2.0)
                                  .build());

    auto house = mesh_loader.load("/home/emil/models/house1/black_smith.obj");
    /*entities.emplace_back(
            house[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 90.0, 0.0})
                    .with_uniform_scale(100.0f)
                    .build()
    );*/

    auto dragon = mesh_loader.load("/home/emil/models/stanford_dragon/dragon.obj");
    dragon[0]->set_material(paper_material);
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
   /* entities.emplace_back(light_mesh[0],
                          WorldTransformBuilder()
                                  .with_translation({0.0, 960.0, 0.0})
                                  .with_scale({10.0, 0.1, 10.0})
                                  .build());
*/

    auto lamp_height = 150.0f;
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

    for (int i = 0; i < 4; ++i) {
        entities.emplace_back(
                wall_lamp[i],
                WorldTransformBuilder()
                        .with_object_space_translation({0.0, -2.0246, 0.0})
                        .with_translation({0.0, lamp_height, 515.0})
                        .with_rotation({0.0, glm::pi<float>(), 0.0f})
                        .with_uniform_scale(300.0f)
                        .build()
        );
    }


    // Back wall
    entities.emplace_back(
            wall_mesh2[0],
            WorldTransformBuilder()
                    .with_translation({0.0, 480.0, -480.0})
                    .with_rotation({glm::pi<float>() / 2.0f, 0.0, -glm::pi<float>() / 2.0f})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );
    for (int i = 0; i < 4; ++i) {
        entities.emplace_back(
                wall_lamp[i],
                WorldTransformBuilder()
                        .with_object_space_translation({0.0, -2.0246, 0.0})
                        .with_translation({0.0, lamp_height, -505.0})
                        .with_rotation({0.0, 0.0, 0.0f})
                        .with_uniform_scale(300.0f)
                        .build()
        );
    }

    // Left wall
    entities.emplace_back(
            wall_mesh[0],
            WorldTransformBuilder()
                    .with_translation({-480.0, 480.0, 0.0})
                    .with_rotation({0.0f, -glm::pi<float>() / 2.0f, glm::pi<float>() / 2.0f})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );
    for (int i = 0; i < 4; ++i) {
        entities.emplace_back(
                wall_lamp[i],
                WorldTransformBuilder()
                        .with_object_space_translation({0.0, -2.0246, 0.0})
                        .with_translation({-515.0, lamp_height, 0.0f})
                        .with_rotation({0.0, glm::pi<float>() / 2.0, 0.0f})
                        .with_uniform_scale(300.0f)
                        .build()
        );
    }

    // Right wall
    entities.emplace_back(
            wall_mesh[0],
            WorldTransformBuilder()
                    .with_translation({480.0, 480.0, 0.0})
                    .with_rotation({0.0, glm::pi<float>() / 2.0f, -glm::pi<float>() / 2.0f})
                    .with_scale({10.0, 0.1, 10.0})
                    .build()
    );

    for (int i = 0; i < 4; ++i) {
        entities.emplace_back(
                wall_lamp[i],
                WorldTransformBuilder()
                        .with_object_space_translation({0.0, -2.0246, 0.0})
                        .with_translation({515.0, lamp_height, 0.0f})
                        .with_rotation({0.0, -glm::pi<float>() / 2.0, 0.0f})
                        .with_uniform_scale(300.0f)
                        .build()
        );
    }
}

size_t sample = 0;
Scene *scene;
DeviceTextureLoader texture_loader;
void drop_callback(GLFWwindow* window, int count, const char** paths)
{
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    for (auto i = 0;  i < count;  i++) {
        if(DeviceTextureLoader::file_is_supported(paths[i])) {
            auto texture = texture_loader.load(paths[i]);
            if(texture) {
                scene->set_sky_texture(texture);
                sample = 0;
            }
        }
    }
}

int main() {
    init_glfw();
    std::cout << "size is: " << (sizeof(glm::vec3) * 4096 * 4096) / (1024*1024) << std::endl;
    std::cout << "size is: " << (sizeof(uint8_t) * 4096 * 4096) / (1024*1024) << std::endl;
    //return 1;
    GlWindow window{"CUDA Raytracer", WIDTH, HEIGHT, keyboard_func};

    glfwSetDropCallback(window.handle(), drop_callback);


    init_gl_buffers();

    print_cuda_device_info();

    Renderer rend{opengl_tex_cuda, WIDTH, HEIGHT};

    auto camera = Camera::create();

    float rot = 1.45f;
    //auto camera_position = glm::vec3(glm::cos(rot) * 10.0, 0.0000, glm::sin(rot) * 10.0f);
    // auto camera_position = glm::vec3(90.0, 100.0, 200.0);
    auto camera_position = glm::vec3(200.0, 50.0, 200.0);
    auto camera_direction = glm::normalize(glm::vec3(0.0, 0.0, 0.0f) - camera_position);
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

    cudaDeviceSetLimit(cudaLimitStackSize, 2048);

    glfwPollEvents();
    glfwPollEvents();

    DeviceMaterialLoader material_loader{texture_loader};
    std::vector<SceneEntity> entities;
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    //scene_wall_lamps(mesh_loader, material_loader, texture_loader, entities);
    // scene_house(mesh_loader, material_loader, texture_loader, entities);// gen);
    scene_dragon(mesh_loader, material_loader, texture_loader, entities, gen);


    cudaMallocManaged(&scene, sizeof(Scene));
    new(scene) Scene;
    scene->build(entities);

    //auto sky = texture_loader.load("/home/emil/textures/sky.png");
    auto sky = texture_loader.load("/home/emil/textures/sunset2.jpg");
    //auto sky = texture_loader.load("/home/emil/textures/red.png");
    scene->set_sky_texture(sky);

    std::cout << "Creating random states..." << std::flush;
    auto random = create_device_type<RandomGeneratorPool>(2048 * 256, 682856);
    std::cout << "Done." << std::endl;
    double rotation = 0.0;
    double total_duration = 0.0f;
    double max_duration = 0.0f;
    int frame_counter = 0;

    glfwSetInputMode(window.handle(), GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    glfwSetMouseButtonCallback(window.handle(), mouse_button_callback);


    auto run = true;

    float yaw = glm::pi<float>() * 0.75f;
    float pitch = 0.0f;

    set_camera_direction(camera, yaw, pitch);
    device_autofocus(camera, scene, WIDTH, HEIGHT);
    while (run && !window.should_close()) {
        handle_input(window, camera, scene);

        if (glfwGetKey(window.handle(), GLFW_KEY_ESCAPE)) {
            run = false;
        }

        if (glfwGetKey(window.handle(), GLFW_KEY_P)) {
            sample = 0;
        }
        if (glfwGetKey(window.handle(), GLFW_KEY_O)) {
            sample = 0;
        }

        if (glfwGetKey(window.handle(), GLFW_KEY_Y)) {
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

        if (needs_autofocus) {
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
        /*if (sample > 3) {
            run = false;
        }*/
    }

    cudaFree(camera);

    // https://stackoverflow.com/questions/14446495/cmake-project-structure-with-unit-tests
    // https://bitbucket.org/EmilNorden/physicstracer/src/master/CMakeLists.txt
    return 0;
}
