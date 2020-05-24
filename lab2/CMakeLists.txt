cmake_minimum_required(VERSION 3.16)

project(CUBLAS LANGUAGES CUDA CXX)

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
set(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
set(LIBS ${LIBS} ${ALL_CUDA_LIBS})

add_executable(CUBLAS matrixMulCUBLAS.cpp)
add_executable(STANDARD matrixMultiplication.cu)
target_link_libraries(CUBLAS ${LIBS})