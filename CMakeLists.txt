cmake_minimum_required(VERSION 3.10)
project(_compliant_gym_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIRS})

add_executable(cylinder_visualizer
        src/cylinder_visualizer.cpp)
target_link_libraries(cylinder_visualizer PUBLIC raisim::raisim pthread)
target_include_directories(cylinder_visualizer PUBLIC ${EIGEN3_INCLUDE_DIRS} include module)
if (CMAKE_BUILD_TYPE STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cylinder_visualizer PRIVATE -mtune=native -fPIC -O3 -g -march=native)
endif()