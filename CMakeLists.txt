cmake_minimum_required(VERSION 3.11.0)
project(ground_texture_slam_new 
  VERSION 1.0.2
  DESCRIPTION "A SLAM system that only uses the ground beneath the robot"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
# Provides default install locations for Ubuntu
include(GNUInstallDirs)

find_package(GTSAM 4.0.3 REQUIRED)
find_package(OpenCV 4.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include(FetchContent)
# Add support for Python bindings
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.9.1
  GIT_SHALLOW true
)
# Add support for DBoW2
FetchContent_Declare(
  DBoW2
  GIT_REPOSITORY https://github.com/dorian3d/DBoW2.git
  GIT_TAG master
  GIT_SHALLOW true
)
# Turn off DBoW2's demo before adding to reduce things to compile.
set(BUILD_Demo OFF CACHE BOOL "Whether to build DBoW2's demo")
FetchContent_MakeAvailable(pybind11 DBoW2)
unset(BUILD_DEMO)

if (BUILD_TESTING)
  # Add flags to emit coverage data
  # Need to have before compiling source files
  add_compile_options("--coverage")
  add_link_options("--coverage")
  # needed for clang-tidy
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif (BUILD_TESTING)

# Add the C++ library
add_library(ground_texture_slam_new
  src/ground_texture_slam_new/BagOfWords.cpp
  src/ground_texture_slam_new/GroundTextureSLAM.cpp
  src/ground_texture_slam_new/ImageParser.cpp
  src/ground_texture_slam_new/KeypointMatcher.cpp
  src/ground_texture_slam_new/TransformEstimator.cpp
)
target_include_directories(ground_texture_slam_new
  PUBLIC
    include/ground_texture_slam_new
    ${DBoW2_SOURCE_DIR}/include
    ${GTSAM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ground_texture_slam_new
  PUBLIC
    DBoW2
    gtsam
    ${OpenCV_LIBS}
)
target_compile_definitions(ground_texture_slam_new
  PUBLIC
    OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
)
install(
  TARGETS
    ground_texture_slam_new
)

pybind11_add_module(ground_texture_slam_new_python
  src/ground_texture_slam_new/BagOfWords.cpp
  src/ground_texture_slam_new/GroundTextureSLAM.cpp
  src/ground_texture_slam_new/ImageParser.cpp
  src/ground_texture_slam_new/KeypointMatcher.cpp
  src/ground_texture_slam_new/TransformEstimator.cpp
  src/ground_texture_slam_new/python_bindings.cpp
)
target_include_directories(ground_texture_slam_new_python
  PUBLIC
    include/ground_texture_slam_new
    ${DBoW2_SOURCE_DIR}/include
    ${GTSAM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ground_texture_slam_new_python
  PUBLIC
    DBoW2
    gtsam
    ${OpenCV_LIBS}
)
target_compile_definitions(ground_texture_slam_new_python
  PUBLIC
    BUILD_PYTHON
    OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
)
set_target_properties(ground_texture_slam_new_python
  PROPERTIES
    OUTPUT_NAME ground_texture_slam_new
)
# Make sure the Python library gets installed where Python actually looks for it. This is
# maybe a bit hacky. Finding Python tells us where the various packages should be installed via
# setting the Python_SITEARCH variable.
find_package(Python REQUIRED Interpreter)
message(STATUS "Python bindings install target: " ${Python_SITEARCH})
install(
  TARGETS ground_texture_slam_new_python
  COMPONENT python
  LIBRARY DESTINATION ${Python_SITEARCH}
)

# Use Google Test for unit testing.
include(CTest)
if(BUILD_TESTING)
  # Prepare Google Test
  FetchContent_Declare(
		google_test
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG main
		GIT_SHALLOW true
	)
	FetchContent_MakeAvailable(google_test)
  include(GoogleTest)
  add_executable(unit_tests
    test/testBagOfWords.cpp
    test/testGroundTextureSLAM.cpp
    test/testImageParser.cpp
    test/testKeypointMatcher.cpp
    test/testTransformEstimator.cpp
  )
  target_include_directories(unit_tests
    PRIVATE
      include
  )
  target_link_libraries(unit_tests
    ground_texture_slam_new
    gtest_main
  )
  gtest_discover_tests(unit_tests)
endif(BUILD_TESTING)

# Include a simple example for demonstration.
add_executable(simple_example
  src/simple_example.cpp
)
target_include_directories(simple_example
  PRIVATE
    include
)
target_link_libraries(simple_example
  PRIVATE
    ground_texture_slam_new
)
install(
  TARGETS
    simple_example
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
