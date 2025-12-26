add_executable(unit_tests 
    ut_tensor.cpp
    ut_nn.cpp
    ut_dataset.cpp
)

target_link_libraries(unit_tests PRIVATE
    core_lib
    GTest::gtest
    GTest::gtest_main
)


target_include_directories(unit_tests PUBLIC ${CMAKE_SOURCE_DIR}/include)

gtest_discover_tests(unit_tests)