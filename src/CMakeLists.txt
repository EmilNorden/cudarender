set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${renderer_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DRENDER_DEBUG")

add_library(renderer
        renderer/aabb.cuh
        renderer/autofocus.cu
        renderer/autofocus.cuh
        renderer/renderer.cu
        renderer/renderer.cuh
        renderer/camera.cu
        renderer/camera.cuh
        renderer/coordinates.cuh
        renderer/cuda_utils.cu
        renderer/cuda_utils.cuh
        renderer/device_material.cu
        renderer/device_material.cuh
        renderer/device_material_loader.cu
        renderer/device_material_loader.cuh
        renderer/device_mesh_loader.cu
        renderer/device_mesh_loader.cuh
        renderer/device_mesh.cu
        renderer/device_mesh.cuh
        renderer/device_random.cu
        renderer/device_random.cuh
        renderer/device_stack.cuh
        renderer/device_texture.cu
        renderer/device_texture.cuh
        renderer/device_texture_loader.cu
        renderer/device_texture_loader.cuh
        renderer/geometry_helpers.cu
        renderer/geometry_helpers.cuh
        renderer/intersection.cuh
        renderer/kd_tree.cuh
        renderer/ray.cuh
        renderer/scene.cuh
        renderer/scene_entity.cu
        renderer/scene_entity.cuh
        renderer/transform.cuh
        )

# find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(assimp REQUIRED)
find_package(FreeImage REQUIRED)

set(OptiX_INSTALL_DIR /home/emil/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64)
message(STATUS "hellosss ${OptiX_INSTALL_DIR}")

find_package(OptiX REQUIRED)

### RENDERER

target_include_directories(renderer SYSTEM PRIVATE
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${ASSIMP_INCLUDE_DIRS}
        ${FREEIMAGE_INCLUDE_DIRS})

target_link_libraries(renderer
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
        ${ASSIMP_LIBRARIES}
        ${FREEIMAGE_LIBRARIES}
        glfw)

set_target_properties(
        renderer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(renderer PUBLIC RENDERER_CUDA_BUILD)


### BINARY

add_executable(renderer_bin
        main.cu
        kernel.cu
        content/model.cpp
        content/model.h
        content/model_loader.cpp
        content/model_loader.h
        shader_tools/GLSLProgram.cpp
        shader_tools/GLSLShader.cpp
        gui/gl_window.cpp
        gui/gl_window.h
        gui/glew_helper.cpp
        gui/glew_helper.h)

target_include_directories(renderer_bin SYSTEM PRIVATE
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${ASSIMP_INCLUDE_DIRS})

target_link_libraries(renderer_bin
        renderer
        ${ASSIMP_LIBRARIES}
        )

set_target_properties(
        renderer_bin
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(renderer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        -arch=compute_86
        -code=sm_86
        >)