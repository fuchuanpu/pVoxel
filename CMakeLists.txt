cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(pVoxel)
set(CMAKE_CXX_STANDARD 14)

aux_source_directory(. DIR_SRCS)
add_executable(${PROJECT_NAME} "${DIR_SRCS}" ./common.hpp ./json.hpp)

add_compile_options(-fopenmp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME} gflags)
target_link_libraries(${PROJECT_NAME} boost_filesystem)

target_link_libraries(${PROJECT_NAME}
    armadillo
    mlpack
)