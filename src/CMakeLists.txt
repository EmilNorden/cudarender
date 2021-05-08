set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DRENDER_DEBUG")

add_library(renderer
        dummy.cpp
        dummy.h)

# find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

target_include_directories(renderer PRIVATE
        # ${SDL2_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR})

target_link_libraries(renderer
        # ${SDL2_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES}
        glfw)

add_executable(renderer_bin
        main.cu
        kernel.cu
        shader_tools/GLSLProgram.cpp
        shader_tools/GLSLShader.cpp)

target_include_directories(renderer_bin PRIVATE
        # ${SDL2_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR})

target_link_libraries(renderer_bin renderer)

set_target_properties(
        renderer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        renderer_bin
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)