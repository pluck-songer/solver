# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/song/code/gauss_newton_solver

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/song/code/gauss_newton_solver/build

# Include any dependencies generated for this target.
include solver/CMakeFiles/gauss_newton.dir/depend.make

# Include the progress variables for this target.
include solver/CMakeFiles/gauss_newton.dir/progress.make

# Include the compile flags for this target's objects.
include solver/CMakeFiles/gauss_newton.dir/flags.make

solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o: solver/CMakeFiles/gauss_newton.dir/flags.make
solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o: ../solver/gauss_newton.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/song/code/gauss_newton_solver/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o"
	cd /home/song/code/gauss_newton_solver/build/solver && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o -c /home/song/code/gauss_newton_solver/solver/gauss_newton.cpp

solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gauss_newton.dir/gauss_newton.cpp.i"
	cd /home/song/code/gauss_newton_solver/build/solver && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/song/code/gauss_newton_solver/solver/gauss_newton.cpp > CMakeFiles/gauss_newton.dir/gauss_newton.cpp.i

solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gauss_newton.dir/gauss_newton.cpp.s"
	cd /home/song/code/gauss_newton_solver/build/solver && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/song/code/gauss_newton_solver/solver/gauss_newton.cpp -o CMakeFiles/gauss_newton.dir/gauss_newton.cpp.s

solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.requires:

.PHONY : solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.requires

solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.provides: solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.requires
	$(MAKE) -f solver/CMakeFiles/gauss_newton.dir/build.make solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.provides.build
.PHONY : solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.provides

solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.provides.build: solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o


# Object files for target gauss_newton
gauss_newton_OBJECTS = \
"CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o"

# External object files for target gauss_newton
gauss_newton_EXTERNAL_OBJECTS =

solver/libgauss_newton.a: solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o
solver/libgauss_newton.a: solver/CMakeFiles/gauss_newton.dir/build.make
solver/libgauss_newton.a: solver/CMakeFiles/gauss_newton.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/song/code/gauss_newton_solver/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libgauss_newton.a"
	cd /home/song/code/gauss_newton_solver/build/solver && $(CMAKE_COMMAND) -P CMakeFiles/gauss_newton.dir/cmake_clean_target.cmake
	cd /home/song/code/gauss_newton_solver/build/solver && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gauss_newton.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
solver/CMakeFiles/gauss_newton.dir/build: solver/libgauss_newton.a

.PHONY : solver/CMakeFiles/gauss_newton.dir/build

solver/CMakeFiles/gauss_newton.dir/requires: solver/CMakeFiles/gauss_newton.dir/gauss_newton.cpp.o.requires

.PHONY : solver/CMakeFiles/gauss_newton.dir/requires

solver/CMakeFiles/gauss_newton.dir/clean:
	cd /home/song/code/gauss_newton_solver/build/solver && $(CMAKE_COMMAND) -P CMakeFiles/gauss_newton.dir/cmake_clean.cmake
.PHONY : solver/CMakeFiles/gauss_newton.dir/clean

solver/CMakeFiles/gauss_newton.dir/depend:
	cd /home/song/code/gauss_newton_solver/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/song/code/gauss_newton_solver /home/song/code/gauss_newton_solver/solver /home/song/code/gauss_newton_solver/build /home/song/code/gauss_newton_solver/build/solver /home/song/code/gauss_newton_solver/build/solver/CMakeFiles/gauss_newton.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : solver/CMakeFiles/gauss_newton.dir/depend

