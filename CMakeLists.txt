cmake_minimum_required(VERSION 3.22)
project(slime_mold CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


find_library(rlib raylib REQUIRED)


set(CMAKE_CUDA_STANDARD 14)

add_executable(slime_mold main.cu lodepng.cpp lodepng.h)


target_link_libraries(slime_mold raylib)


set_target_properties(slime_mold PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
