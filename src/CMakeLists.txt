
set(CORE_SOURCES
    tensor.cpp
    neural_network.cpp
    dataset.cpp
)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE core_lib)

add_library(core_lib STATIC ${CORE_SOURCES})


target_include_directories(core_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/include)