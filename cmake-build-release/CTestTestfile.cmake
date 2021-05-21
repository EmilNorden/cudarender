# CMake generated Testfile for 
# Source directory: /home/emil/code/cuda-test2
# Build directory: /home/emil/code/cuda-test2/cmake-build-release
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[MyTest]=] "/home/emil/code/cuda-test2/cmake-build-release/test/Test")
set_tests_properties([=[MyTest]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/emil/code/cuda-test2/CMakeLists.txt;9;add_test;/home/emil/code/cuda-test2/CMakeLists.txt;0;")
subdirs("src")
subdirs("test")
