cmake_minimum_required(VERSION 3.20)

project(integral_image)

add_executable(integral_image integral_image_cli.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(integral_image boost_program_options ${OpenCV_LIBS})