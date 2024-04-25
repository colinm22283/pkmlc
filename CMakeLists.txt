cmake_minimum_required(VERSION 3.27)
project(pkmlc)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

add_executable(pkmlc
        include/parser.hpp
        include/module/module.hpp
)

FetchContent_Declare(
        json
        GIT_REPOSITORY git@github.com:nlohmann/json.git
        GIT_TAG        v3.11.3
)

FetchContent_MakeAvailable(json)

target_include_directories(pkmlc PUBLIC include)

add_subdirectory(source)

target_link_libraries(pkmlc PUBLIC nlohmann_json)
