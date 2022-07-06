cmake_minimum_required(VERSION 3.10)
project(controller LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIRS})

subdir_names(CONTROLLERS .)
list(REMOVE_ITEM CONTROLLERS common)

function(create_controller name)
    add_library(${name} STATIC ${name}/${name}.cpp)
    target_link_libraries(${name} raisim::raisim pthread)
    target_include_directories(${name} PUBLIC ${EIGEN3_INCLUDE_DIRS} common)
    if (CMAKE_BUILD_TYPE STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${name} PRIVATE -mtune=native -fPIC -O3 -g -march=native)
    endif()
endfunction()

foreach(con ${CONTROLLERS})
    create_controller(${con})
    message(-- complete: ${con})
endforeach()
