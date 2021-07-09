cmake_minimum_required(VERSION 3.20)

project(integral_image)

add_executable(integral_image integral_image_cli.cpp)
target_link_libraries(integral_image boost_program_options)