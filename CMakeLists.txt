cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 17)
project(NeuralNetwork VERSION 1.0)

find_package(GTest REQUIRED)
add_compile_definitions(
  $<$<CONFIG:Debug>:ENABLE_DEBUG>
)

add_subdirectory(src)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


enable_testing()
add_subdirectory(tests) 
include(GoogleTest)