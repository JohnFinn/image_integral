cmake_minimum_required(VERSION 3.20)

project(integral_image)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(integrate_image integrate_image.cpp)

add_executable(integral_image integral_image_cli.cpp)
target_link_libraries(integral_image integrate_image boost_program_options pthread ${OpenCV_LIBS})

add_executable(test_integral_image test_integral_image.cpp)
target_link_libraries(test_integral_image integrate_image gtest pthread ${OpenCV_LIBS})

add_executable(benchmark_integrate_image bm_integrate_image.cpp)
target_link_libraries(benchmark_integrate_image integrate_image pthread benchmark ${OpenCV_LIBS})