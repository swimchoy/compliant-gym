cmake_minimum_required(VERSION 3.10)
project(_compliant_gym_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "~/Workspace/Libraries/raisimLib/raisim_build")
list(APPEND CMAKE_PREFIX_PATH "~/Workspace/raisin_ws/src/raisin/raisin_torch/pytorch/libtorch")
list(APPEND CMAKE_PREFIX_PATH "~/Workspace/Libraries/libtorch_cpu/libtorch")

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})

include_directories (${EIGEN3_INCLUDE_DIRS})
# file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/rsc" DESTINATION ${CMAKE_BINARY_DIR})

add_executable(cylinder_visualizer
        src/cylinder_visualizer.cpp)
target_link_libraries(cylinder_visualizer PUBLIC raisim::raisim pthread ${TORCH_LIBRARIES})
target_include_directories(cylinder_visualizer PUBLIC ${EIGEN3_INCLUDE_DIRS} include module)
if (CMAKE_BUILD_TYPE STREQUAL "RELEASE" OR CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cylinder_visualizer PRIVATE -mtune=native -fPIC -O3 -g -march=native)
endif()