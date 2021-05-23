add_executable(Test test.cu)

message("-----------> ${renderer_SOURCE_DIR}")

target_include_directories(Test PRIVATE
        ${renderer_SOURCE_DIR}/src)
target_link_libraries(
        Test
        renderer)