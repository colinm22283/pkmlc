cmake_minimum_required(VERSION 3.27)
project(pkmlc)

set(CMAKE_CXX_STANDARD 20)

add_executable(pkmlc
        examples/xor/main.cu
        include/parser.hpp
        include/module/module.hpp
        examples/modules/sigmoid.hpp
        examples/xor/output.hpp)

target_include_directories(pkmlc PUBLIC include)

add_subdirectory(source)
