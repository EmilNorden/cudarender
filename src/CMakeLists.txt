set(CMAKE_CUDA_STANDARD 17)

add_library(renderer
        window.cpp
        window.h)

find_package(SDL2 REQUIRED)
target_include_directories(renderer PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(renderer ${SDL2_LIBRARIES})

add_executable(renderer_bin main.cu)

# This shouldnt be here. I should remove all SDL2 dependencies from main
target_include_directories(renderer_bin PRIVATE ${SDL2_INCLUDE_DIRS})

target_link_libraries(renderer_bin renderer)

set_target_properties(
        renderer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        renderer_bin
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)