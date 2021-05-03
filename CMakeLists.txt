cmake_minimum_required(VERSION 3.19)
project(renderer CUDA CXX)

# add_executable(cuda_test2 src/main.cu src/window.cpp src/window.h)

add_subdirectory(src)
add_subdirectory(test)
enable_testing()
add_test(NAME MyTest COMMAND Test)