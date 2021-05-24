# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/emil/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/211.7142.21/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/emil/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/211.7142.21/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/emil/code/cuda-test2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/emil/code/cuda-test2/cmake-build-release

# Include any dependencies generated for this target.
include test/CMakeFiles/Test.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/Test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/Test.dir/flags.make

test/CMakeFiles/Test.dir/test.cu.o: test/CMakeFiles/Test.dir/flags.make
test/CMakeFiles/Test.dir/test.cu.o: ../test/test.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/emil/code/cuda-test2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test/CMakeFiles/Test.dir/test.cu.o"
	cd /home/emil/code/cuda-test2/cmake-build-release/test && /usr/local/cuda-11.3/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/emil/code/cuda-test2/test/test.cu -o CMakeFiles/Test.dir/test.cu.o

test/CMakeFiles/Test.dir/test.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Test.dir/test.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/CMakeFiles/Test.dir/test.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Test.dir/test.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target Test
Test_OBJECTS = \
"CMakeFiles/Test.dir/test.cu.o"

# External object files for target Test
Test_EXTERNAL_OBJECTS =

test/CMakeFiles/Test.dir/cmake_device_link.o: test/CMakeFiles/Test.dir/test.cu.o
test/CMakeFiles/Test.dir/cmake_device_link.o: test/CMakeFiles/Test.dir/build.make
test/CMakeFiles/Test.dir/cmake_device_link.o: src/librenderer.a
test/CMakeFiles/Test.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libGLEW.so
test/CMakeFiles/Test.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libOpenGL.so
test/CMakeFiles/Test.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libGLX.so
test/CMakeFiles/Test.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libGLU.so
test/CMakeFiles/Test.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libfreeimage.so
test/CMakeFiles/Test.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
test/CMakeFiles/Test.dir/cmake_device_link.o: test/CMakeFiles/Test.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/emil/code/cuda-test2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/Test.dir/cmake_device_link.o"
	cd /home/emil/code/cuda-test2/cmake-build-release/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Test.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/Test.dir/build: test/CMakeFiles/Test.dir/cmake_device_link.o

.PHONY : test/CMakeFiles/Test.dir/build

# Object files for target Test
Test_OBJECTS = \
"CMakeFiles/Test.dir/test.cu.o"

# External object files for target Test
Test_EXTERNAL_OBJECTS =

test/Test: test/CMakeFiles/Test.dir/test.cu.o
test/Test: test/CMakeFiles/Test.dir/build.make
test/Test: src/librenderer.a
test/Test: /usr/lib/x86_64-linux-gnu/libGLEW.so
test/Test: /usr/lib/x86_64-linux-gnu/libOpenGL.so
test/Test: /usr/lib/x86_64-linux-gnu/libGLX.so
test/Test: /usr/lib/x86_64-linux-gnu/libGLU.so
test/Test: /usr/lib/x86_64-linux-gnu/libfreeimage.so
test/Test: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
test/Test: test/CMakeFiles/Test.dir/cmake_device_link.o
test/Test: test/CMakeFiles/Test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/emil/code/cuda-test2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable Test"
	cd /home/emil/code/cuda-test2/cmake-build-release/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/Test.dir/build: test/Test

.PHONY : test/CMakeFiles/Test.dir/build

test/CMakeFiles/Test.dir/clean:
	cd /home/emil/code/cuda-test2/cmake-build-release/test && $(CMAKE_COMMAND) -P CMakeFiles/Test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/Test.dir/clean

test/CMakeFiles/Test.dir/depend:
	cd /home/emil/code/cuda-test2/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/emil/code/cuda-test2 /home/emil/code/cuda-test2/test /home/emil/code/cuda-test2/cmake-build-release /home/emil/code/cuda-test2/cmake-build-release/test /home/emil/code/cuda-test2/cmake-build-release/test/CMakeFiles/Test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/Test.dir/depend
