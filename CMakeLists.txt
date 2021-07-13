cmake_minimum_required(VERSION 3.20)

project(integral_image)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(lib_integrate_image integrate_image.cpp)

add_executable(integral_image integral_image_cli.cpp)
target_link_libraries(integral_image lib_integrate_image boost_program_options ${OpenCV_LIBS})

add_executable(test_integral_image test_integral_image.cpp)
target_link_libraries(test_integral_image lib_integrate_image gtest ${OpenCV_LIBS})