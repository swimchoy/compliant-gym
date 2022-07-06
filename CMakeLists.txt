cmake_minimum_required(VERSION 3.10)
project(_compliant_gym_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIRS})

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/rsc" DESTINATION ${CMAKE_BINARY_DIR})

function(create_executable name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PUBLIC raisim::raisim pthread)
    target_include_directories(${name} PUBLIC ${EIGEN3_INCLUDE_DIRS} include module)
    if (CMAKE_BUILD_TYPE STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${name} PRIVATE -mtune=native -fPIC -O3 -g -march=native)
    endif()
endfunction()

create_executable(cylinder_visualizer)