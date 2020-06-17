project(Project)

cmake_minimum_required(VERSION 3.15)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
enable_testing() # needed on top-level CMakeLists.txt


add_subdirectory(lab1)
add_subdirectory(lab2)
add_subdirectory(lab3)